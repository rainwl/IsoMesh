using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEditor;
using UnityEditor.Compilation;

namespace IsoMesh
{
    /// <summary>
    /// An SDF group is a collection of sdf primitives, meshes, and operations which mutually interact.
    /// This class is not responsible for rendering the result or indeed doing anything with it. Instead, it dispatches
    /// the resulting buffers to SDF Components. These components must be a child of the group and implement the ISDFComponent interface.
    /// 
    /// This class also contains the functionality to directly provide information about the SDF, like raycasting.
    /// </summary>
    [ExecuteInEditMode]
    public class SDFGroup : MonoBehaviour
    {
        #region Fields and Properties

        private static class GlobalProperties
        {
            public static int MeshSamples_StructuredBuffer = Shader.PropertyToID("_SDFMeshSamples");
            public static int MeshPackedUVs_StructuredBuffer = Shader.PropertyToID("_SDFMeshPackedUVs");
        }

        public const float MIN_SMOOTHING = 0.00001f;

        /// <summary>
        /// Whether this group is actively updating.
        /// </summary>
        public bool IsRunning => m_isRunning;

        [SerializeField]
        private bool m_isRunning = true;

        private bool m_forceUpdateNextFrame = false;

        /// <summary>
        /// Whether this group is fully set up, e.g. all the buffers are created.
        /// </summary>
        public bool IsReady { get; private set; } = false;

        [SerializeField]
        [HideInInspector]
        // this bool is toggled off/on whenever the Unity callbacks OnEnable/OnDisable are called.
        // note that this doesn't always give the same result as "enabled" because OnEnable/OnDisable are
        // called during recompiles etc. you can basically read this bool as "is recompiling"
        private bool m_isEnabled = false;

        [SerializeField]
        private float m_normalSmoothing = 0.015f;
        public float NormalSmoothing => m_normalSmoothing;

        [SerializeField]
        private float m_thicknessMaxDistance = 0f;

        [SerializeField]
        private float m_thicknessFalloff = 0f;

        private List<ISDFGroupComponent> m_sdfComponents = new List<ISDFGroupComponent>();

        private ComputeBuffer m_dataBuffer;
        private ComputeBuffer m_materialBuffer;

        private ComputeBuffer m_settingsBuffer;
        public ComputeBuffer SettingsBuffer => m_settingsBuffer;

        private Settings[] m_settingsArray = new Settings[1];

        private List<SDFObject> m_sdfObjects = new List<SDFObject>();

        private static readonly List<SDFMesh> m_globalSDFMeshes = new List<SDFMesh>();
        private static readonly Dictionary<int, int> m_meshSdfSampleStartIndices = new Dictionary<int, int>();
        private static readonly Dictionary<int, int> m_meshSdfUVStartIndices = new Dictionary<int, int>();
        private static readonly Dictionary<int, int> m_meshCounts = new Dictionary<int, int>();
        private static readonly List<float> m_meshSamples = new List<float>();
        private static readonly List<float> m_meshPackedUVs = new List<float>();

        private static ComputeBuffer m_meshSamplesBuffer;
        private static ComputeBuffer m_meshPackedUVsBuffer;

        private List<SDFGPUData> m_data = new List<SDFGPUData>();
        private List<SDFMaterialGPU> m_materials = new List<SDFMaterialGPU>();
        private readonly List<int> m_dataSiblingIndices = new List<int>();

        public bool IsEmpty => m_sdfObjects.IsNullOrEmpty();

        // dirty flags. we only need one for the primitives, but two for the meshes. this is because
        // i want to avoid doing a 'full update' of the meshes unless i really need to.

        private static bool m_isGlobalMeshDataDirty = true;
        private bool m_isLocalDataDirty = true;
        private bool m_isLocalDataOrderDirty = true;

        /// <summary>
        /// The mapper allows you to quickly query the SDF without involving the GPU.
        /// </summary>
        public Mapper Mapper { get; } = new Mapper();

        #endregion

        #region Registration

        public void Register(SDFObject sdfObject)
        {
            if (m_sdfObjects.Contains(sdfObject))
                return;

            if (sdfObject is SDFMesh sdfMesh)
            {
                // check if this is a totally new mesh that no group has seen
                if (!m_globalSDFMeshes.Contains(sdfMesh))
                {
                    m_globalSDFMeshes.Add(sdfMesh);
                    m_isGlobalMeshDataDirty = true;
                }

                // keep track of how many groups contain a local reference to this sdfmesh
                if (!m_meshCounts.ContainsKey(sdfMesh.ID))
                    m_meshCounts.Add(sdfMesh.ID, 0);

                m_meshCounts[sdfMesh.ID]++;
            }

            bool wasEmpty = IsEmpty;

            m_sdfObjects.Add(sdfObject);
            m_isLocalDataDirty = true;
            m_isLocalDataOrderDirty = true;

            // this is almost certainly overkill, but i like the kind of guaranteed stability
            ClearNulls(m_sdfObjects);

            if (wasEmpty && !IsEmpty)
                for (int i = 0; i < m_sdfComponents.Count; i++)
                    m_sdfComponents[i].OnNotEmpty();

            RequestUpdate();
        }

        public void Deregister(SDFObject sdfObject)
        {
            bool wasEmpty = IsEmpty;

            if (!m_sdfObjects.Remove(sdfObject))
                return;

            if (sdfObject is SDFMesh sdfMesh)
            {
                // if this was the only group referencing this sdfmesh, we can remove it from the global buffer too
                if (m_meshCounts.ContainsKey(sdfMesh.ID))
                {
                    m_meshCounts[sdfMesh.ID]--;

                    if (m_meshCounts[sdfMesh.ID] <= 0 && m_globalSDFMeshes.Remove(sdfMesh))
                        m_isGlobalMeshDataDirty = true;
                }
            }

            m_isLocalDataDirty = true;
            m_isLocalDataOrderDirty = true;

            // this is almost certainly overkill
            ClearNulls(m_sdfObjects);

            if (!wasEmpty && IsEmpty)
                for (int i = 0; i < m_sdfComponents.Count; i++)
                    m_sdfComponents[i].OnEmpty();

            RequestUpdate();
        }

        public bool IsRegistered(SDFObject sdfObject) => !m_sdfObjects.IsNullOrEmpty() && m_sdfObjects.Contains(sdfObject);

        #endregion

        #region MonoBehaviour Callbacks

        private void OnEnable()
        {
#if UNITY_EDITOR
            CompilationPipeline.compilationStarted += OnCompilationStarted;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
#endif

            m_isEnabled = true;
            m_isGlobalMeshDataDirty = true;
            m_isLocalDataDirty = true;
            m_isLocalDataOrderDirty = true;

            RequestUpdate(onlySendBufferOnChange: false);
            m_forceUpdateNextFrame = true;
        }

        private void Start()
        {
            m_isEnabled = true;
            m_isGlobalMeshDataDirty = true;
            m_isLocalDataDirty = true;
            m_isLocalDataOrderDirty = true;

            RequestUpdate(onlySendBufferOnChange: false);
            m_forceUpdateNextFrame = true;
        }

        private void OnDisable()
        {
#if UNITY_EDITOR
            CompilationPipeline.compilationStarted -= OnCompilationStarted;
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
#endif

            m_isEnabled = false;
            IsReady = false;

            m_dataBuffer?.Dispose();
            m_materialBuffer?.Dispose();
            m_settingsBuffer?.Dispose();
        }

        private void OnApplicationQuit()
        {
            // static buffers can't be cleared in ondisable or something,
            // because lots of objects might be using them
            m_isEnabled = false;

            m_meshSamplesBuffer?.Dispose();
            m_meshPackedUVsBuffer?.Dispose();
        }

        private void LateUpdate()
        {
            if (!m_isRunning)
                return;

            if (!IsReady)
                RequestUpdate();

            bool nullHit = false;
            for (int i = 0; i < m_sdfObjects.Count; i++)
            {
                SDFObject sdfObject = m_sdfObjects[i];
                bool isNull = !sdfObject;

                nullHit |= isNull;

                if (!isNull)
                {
                    m_isLocalDataDirty |= sdfObject.IsDirty;
                    m_isLocalDataOrderDirty |= sdfObject.IsOrderDirty;
                }
            }

            if (nullHit)
                ClearNulls(m_sdfObjects);

            bool changed = false;

            if (m_isLocalDataOrderDirty)
            {
                ReorderObjects();
                changed = true;
            }

            if (changed || m_forceUpdateNextFrame || m_isGlobalMeshDataDirty || m_isLocalDataDirty || transform.hasChanged)
            {
                changed = true;
                RebuildData();
            }

            m_forceUpdateNextFrame = false;
            transform.hasChanged = false;

            if (changed && !IsEmpty)
            {
                for (int i = 0; i < m_sdfComponents.Count; i++)
                    m_sdfComponents[i].Run();
            }
        }

        #endregion

        #region Buffer Updating

        /// <summary>
        /// Request a complete buffer rebuild.
        /// </summary>
        /// <param name="onlySendBufferOnChange">Whether to invoke the components and inform them the buffer has changed. This is only really necessary when the size changes.</param>
        public void RequestUpdate(bool onlySendBufferOnChange = true)
        {
            if (!m_isEnabled)
                return;

            // blocking readiness because we're updating 
            // all the information at once, we don't want groups to start acting
            // on the info immediately
            IsReady = false;

            m_sdfComponents.Clear();
            m_sdfComponents.AddRange(GetComponentsInChildren<ISDFGroupComponent>());

            if (IsEmpty)
                for (int i = 0; i < m_sdfComponents.Count; i++)
                    m_sdfComponents[i].OnEmpty();
            else
                for (int i = 0; i < m_sdfComponents.Count; i++)
                    m_sdfComponents[i].OnNotEmpty();

            RebuildData(onlySendBufferOnChange);
            OnSettingsChanged();

            IsReady = true;

            if (!IsEmpty)
                for (int i = 0; i < m_sdfComponents.Count; i++)
                    m_sdfComponents[i].Run();
        }

        /// <summary>
        /// Some mesh data is shared across all instances, such as the sample and UV information as well as the start indices in those static buffers
        /// for all meshes. Returns true if the static buffers have been changed and need to be resent to the groups.
        /// 一些网格数据在所有实例之间共享，例如样本和UV信息以及所有网格的静态缓冲区中的开始索引。
        /// 如果静态缓冲区已被更改并且需要重新发送到组，则返回true。
        /// </summary>
        /// <param name="locals">List of SDFMesh objects to ensure are in the global list.</param>
        /// <param name="onlySendBufferOnChange">Whether to invoke the components and inform them the buffer has changed. This is only really necessary when the size changes.
        /// 是否调用组件并通知它们缓冲区已更改。只有当大小发生变化时才需要这样做。
        /// </param>
        /// [AI assist]这段代码是一个名为RebuildGlobalMeshData的方法，主要执行以下操作：
        /// 记录之前的全局网格样本数量(m_meshSamples.Count)和经过打包的UV坐标数量(m_meshPackedUVs.Count)。
        /// 清除全局网格样本、打包的UV坐标、样本的起始索引、UV坐标的起始索引的集合。
        /// 从m_globalSDFMeshes列表中移除为空或其Asset属性为空的SDF网格对象。
        /// 检查传入的locals列表中的每一个元素，如果这个元素是SDF网格并且并未在m_globalSDFMeshes列表中，则将它添加到m_globalSDFMeshes中。
        /// 遍历m_globalSDFMeshes，对于每一个SDF网格，如果它在m_meshCounts中的计数大于0（也就表示它存在于某个群组中），则获取它的样本数据和打包后的UV数据。
        /// 如果这个SDF网格的ID在m_meshSdfSampleStartIndices中不存在，则将网格样本添加到m_meshSamples的末尾，并记录下这些样本在m_meshSamples中的起始索引。类似的处理打包的UV坐标和起始索引。
        /// 如果m_meshSamplesBuffer为null或无效，或者加入新的样本后样本数量发生改变，则释放m_meshSamplesBuffer的内存，创建一个新的buffer，并设置一个标志位表示新的buffer已经创建。
        /// 如果全局样本集合数量大于0，则将其全部数据设置到m_meshSamplesBuffer。
        /// 对打包的UV坐标进行类似步骤7和步骤8的处理，创建新的buffer并设置数据。
        /// 设置全局网格数据状态为“未修改”。
        /// 返回是否创建了新的buffer。
        /// 这段代码整体上的功能是对网格的样本和UV数据进行重新构建和缓存，清除无效的网格，保存有效网格的样本和UV数据，如果有新的数据加入则创建新的计算buffer来存储。
        private static bool RebuildGlobalMeshData(IList<SDFObject> locals, bool onlySendBufferOnChange = true)
        {
            #region Clear

            var previousMeshSamplesCount = m_meshSamples.Count;
            var previousMeshUVsCount = m_meshPackedUVs.Count;

            m_meshSamples.Clear();
            m_meshPackedUVs.Clear();

            m_meshSdfSampleStartIndices.Clear();
            m_meshSdfUVStartIndices.Clear();

            #endregion

            #region Update SDFOjbect

            // remove null refs
            for (var i = m_globalSDFMeshes.Count - 1; i >= 0; --i)
                if (m_globalSDFMeshes[i] == null || m_globalSDFMeshes[i].Asset == null)
                    m_globalSDFMeshes.RemoveAt(i);

            foreach (var sdfObj in locals)
                if (sdfObj is SDFMesh mesh && !m_globalSDFMeshes.Contains(mesh))
                    m_globalSDFMeshes.Add(mesh);

            #endregion


            // loop over each mesh, adding its samples/uvs to the sample buffer
            // and taking note of where each meshes samples start in the buffer.
            // check for repeats so we don't add the same mesh to the samples buffer twice
            // 循环遍历每个网格，将其样本/uv添加到样本缓冲区中，并注意每个网格样本在缓冲区中的起始位置。检查重复，这样我们就不会两次向样本缓冲区添加相同的网格
            foreach (var mesh in m_globalSDFMeshes)
            {
                // ignore meshes which are in the list but not present in any group 忽略列表中但不存在于任何组中的网格
                if (m_meshCounts.TryGetValue(mesh.ID, out var count) && count <= 0)
                    continue;

                mesh.Asset.GetDataArrays(out var samples, out var packedUVs);

                if (!m_meshSdfSampleStartIndices.ContainsKey(mesh.ID))
                {
                    var startIndex = m_meshSamples.Count;
                    m_meshSamples.AddRange(samples);
                    m_meshSdfSampleStartIndices.Add(mesh.ID, startIndex);
                }

                if (mesh.Asset.HasUVs && !m_meshSdfUVStartIndices.ContainsKey(mesh.ID))
                {
                    var startIndex = m_meshPackedUVs.Count;
                    m_meshPackedUVs.AddRange(packedUVs);
                    m_meshSdfUVStartIndices.Add(mesh.ID, startIndex);
                }
            }

            var newBuffers = false;

            if (m_meshSamplesBuffer == null || !m_meshSamplesBuffer.IsValid() || previousMeshSamplesCount != m_meshSamples.Count)
            {
                m_meshSamplesBuffer?.Dispose();
                m_meshSamplesBuffer = new ComputeBuffer(Mathf.Max(1, m_meshSamples.Count), sizeof(float), ComputeBufferType.Structured);
                newBuffers = true;
            }

            if (m_meshSamples.Count > 0)
                m_meshSamplesBuffer.SetData(m_meshSamples);

            if (m_meshPackedUVsBuffer == null || !m_meshPackedUVsBuffer.IsValid() || previousMeshUVsCount != m_meshPackedUVs.Count)
            {
                m_meshPackedUVsBuffer?.Dispose();
                m_meshPackedUVsBuffer = new ComputeBuffer(Mathf.Max(1, m_meshPackedUVs.Count), sizeof(float), ComputeBufferType.Structured);
                newBuffers = true;
            }

            if (m_meshPackedUVs.Count > 0)
                m_meshPackedUVsBuffer.SetData(m_meshPackedUVs);

            m_isGlobalMeshDataDirty = false;

            return newBuffers;
        }

        /// <summary>
        /// Sort the list of sdf objects by sibling index, which ensures that this list is always in the same
        /// order as is shown in the unity hierarchy. This is important because some of the sdf operations are ordered
        /// </summary>
        private void ReorderObjects()
        {
            m_dataSiblingIndices.Clear();

            ClearNulls(m_sdfObjects);

            for (int i = 0; i < m_sdfObjects.Count; i++)
            {
                m_dataSiblingIndices.Add(m_sdfObjects[i].transform.GetSiblingIndex());
                m_sdfObjects[i].SetOrderClean();
            }

            m_sdfObjects = m_sdfObjects.OrderBy(d => m_dataSiblingIndices[m_sdfObjects.IndexOf(d)]).ToList();

            m_isLocalDataOrderDirty = false;
        }

        /// <summary>
        /// Repopulate the data relating to SDF primitives (spheres, toruses, cuboids etc) and SDF meshes (which point to where in the list of sample and uv data they begin, and how large they are)
        /// </summary>
        /// <param name="onlySendBufferOnChange">Whether to invoke the components and inform them the buffer has changed. This is only really necessary when the size changes.</param>
        private void RebuildData(bool onlySendBufferOnChange = true)
        {
            m_isLocalDataDirty = false;

            // should we rebuild the buffers which contain mesh sample + uv data?
            bool globalBuffersChanged = false;
            if (m_meshSamplesBuffer == null || !m_meshSamplesBuffer.IsValid() || m_meshPackedUVsBuffer == null || !m_meshPackedUVsBuffer.IsValid() || m_isGlobalMeshDataDirty)
                globalBuffersChanged = RebuildGlobalMeshData(m_sdfObjects, onlySendBufferOnChange);

            // memorize the size of the array before clearing it, for later comparison
            int previousCount = m_data.Count;
            m_data.Clear();
            m_materials.Clear();

            // add all the sdf objects
            for (int i = 0; i < m_sdfObjects.Count; i++)
            {
                SDFObject sdfObject = m_sdfObjects[i];

                if (!sdfObject)
                    continue;

                sdfObject.SetClean();

                int meshStartIndex = -1;
                int uvStartIndex = -1;

                if (sdfObject is SDFMesh mesh)
                {
                    // get the index in the global samples buffer where this particular mesh's samples begin
                    if (!m_meshSdfSampleStartIndices.TryGetValue(mesh.ID, out meshStartIndex))
                        globalBuffersChanged = RebuildGlobalMeshData(m_sdfObjects, onlySendBufferOnChange); // we don't recognize this mesh so we may need to rebuild the entire global list of mesh samples and UVs

                    // likewise, if this mesh has UVs, get the index where they begin too
                    if (mesh.Asset.HasUVs)
                        m_meshSdfUVStartIndices.TryGetValue(mesh.ID, out uvStartIndex);
                }

                m_data.Add(sdfObject.GetSDFGPUData(meshStartIndex, uvStartIndex));
                m_materials.Add(sdfObject.GetMaterial());
            }

            bool sendBuffer = !onlySendBufferOnChange;

            // check whether we need to create a new buffer. buffers are fixed sizes so the most common occasion for this is simply a change of size
            if (m_dataBuffer == null || !m_dataBuffer.IsValid() || previousCount != m_data.Count)
            {
                sendBuffer = true;

                m_dataBuffer?.Dispose();
                m_dataBuffer = new ComputeBuffer(Mathf.Max(1, m_data.Count), SDFGPUData.Stride, ComputeBufferType.Structured);
            }

            // check whether we need to create a new buffer. buffers are fixed sizes so the most common occasion for this is simply a change of size
            if (m_materialBuffer == null || !m_materialBuffer.IsValid() || previousCount != m_data.Count)
            {
                sendBuffer = true;

                m_materialBuffer?.Dispose();
                m_materialBuffer = new ComputeBuffer(Mathf.Max(1, m_data.Count), SDFMaterialGPU.Stride, ComputeBufferType.Structured);
            }

            // if the buffer is new, the size has changed, or if it's forced, we resend the buffer to the sdf group component classes
            if (sendBuffer)
            {
                for (int i = 0; i < m_sdfComponents.Count; i++)
                    m_sdfComponents[i].UpdateDataBuffer(m_dataBuffer, m_materialBuffer, m_data.Count);
            }

            if (m_data.Count > 0)
            {
                m_dataBuffer.SetData(m_data);
                m_materialBuffer.SetData(m_materials);
            }

            Mapper.SetData(m_data, m_materials);

            // if we also changed the global mesh data buffer in this method, we need to send that as well
            if (!onlySendBufferOnChange || globalBuffersChanged)
            {
                Shader.SetGlobalBuffer(GlobalProperties.MeshSamples_StructuredBuffer, m_meshSamplesBuffer);
                Shader.SetGlobalBuffer(GlobalProperties.MeshPackedUVs_StructuredBuffer, m_meshPackedUVsBuffer);

                Mapper.SetMeshData(m_meshSamples, m_meshPackedUVs);
            }
        }

        #endregion

        #region Events

        /// <summary>
        /// To be called whenever the settings universal to this group change.
        /// </summary>
        public void OnSettingsChanged()
        {
            m_settingsArray[0] = new Settings()
            {
                //Smoothing = Mathf.Max(MIN_SMOOTHING, m_smoothing),
                NormalSmoothing = Mathf.Max(MIN_SMOOTHING, m_normalSmoothing),
                ThicknessMaxDistance = m_thicknessMaxDistance,
                ThicknessFalloff = m_thicknessFalloff,
            };

            if (m_settingsBuffer == null || !m_settingsBuffer.IsValid())
            {
                m_settingsBuffer?.Dispose();
                m_settingsBuffer = new ComputeBuffer(1, Settings.Stride, ComputeBufferType.Structured);
            }

            for (int i = 0; i < m_sdfComponents.Count; i++)
                m_sdfComponents[i].UpdateSettingsBuffer(m_settingsBuffer);

            Mapper.SetSettings(m_settingsArray[0]);

            m_settingsBuffer.SetData(m_settingsArray);
        }

        private void OnCompilationStarted(object param)
        {
            m_isEnabled = false;

            m_meshSamplesBuffer?.Dispose();
            m_meshPackedUVsBuffer?.Dispose();
        }

        private void OnPlayModeStateChanged(PlayModeStateChange stateChange)
        {
            // this ensures "m_isEnabled" is set to false while transitioning between play modes
            m_isEnabled = stateChange == PlayModeStateChange.EnteredPlayMode || stateChange == PlayModeStateChange.EnteredEditMode;

            m_meshSamplesBuffer?.Dispose();
            m_meshPackedUVsBuffer?.Dispose();
        }

        #endregion

        #region Structs

        public struct Settings
        {
            public static int Stride => sizeof(float) * 3;

            public float NormalSmoothing;   // the 'epsilon' value for computing the gradient, affects how smoothed out the normals are
            public float ThicknessMaxDistance;
            public float ThicknessFalloff;

        }

        #endregion

        #region Public Methods

        public Vector3 GetNearestPointOnSurface(Vector3 p) => GetNearestPointOnSurface(p, out _, out _);

        public Vector3 GetNearestPointOnSurface(Vector3 p, out float signedDistance) => GetNearestPointOnSurface(p, out signedDistance, out _);

        public Vector3 GetNearestPointOnSurface(Vector3 p, out float signedDistance, out Vector3 direction)
        {
            signedDistance = Mapper.Map(p);
            direction = -Mapper.MapNormal(p);

            return p + signedDistance * direction;
        }

        public Vector3 GetSurfaceNormal(Vector3 p) => Mapper.MapNormal(p);

        public float GetDistanceToSurface(Vector3 p) => Mapper.Map(p);

        public bool OverlapSphere(Vector3 centre, float radius) => Mapper.Map(centre) <= radius;

        //public bool OverlapBox(Vector3 centre, Vector3 halfExtents, bool surfaceOnly = true) => OverlapBox(centre, halfExtents, Quaternion.identity, surfaceOnly);

        public bool OverlapBox(Vector3 centre, Vector3 halfExtents)
        {
            if (!OverlapSphere(centre, halfExtents.magnitude))
                return false;

            Vector3 maxBounds = centre + halfExtents;
            Vector3 minBounds = centre - halfExtents;

            bool Check(Vector3 p)
            {
                Vector3 surfaceP = GetNearestPointOnSurface(p, out float signedDistance);

                bool isInside =
                    surfaceP.x >= minBounds.x && surfaceP.x <= maxBounds.x &&
                    surfaceP.y >= minBounds.y && surfaceP.y <= maxBounds.y &&
                    surfaceP.z >= minBounds.z && surfaceP.z <= maxBounds.z;

                return isInside;
            }

            return (Check(centre) ||
                Check(centre + halfExtents) ||
                Check(centre - halfExtents) ||
                Check(centre + new Vector3(halfExtents.x, halfExtents.y, -halfExtents.z)) ||
                Check(centre + new Vector3(halfExtents.x, -halfExtents.y, halfExtents.z)) ||
                Check(centre + new Vector3(halfExtents.x, -halfExtents.y, -halfExtents.z)) ||
                Check(centre + new Vector3(-halfExtents.x, halfExtents.y, halfExtents.z)) ||
                Check(centre + new Vector3(-halfExtents.x, halfExtents.y, -halfExtents.z)) ||
                Check(centre + new Vector3(-halfExtents.x, -halfExtents.y, halfExtents.z)));
        }

        /// <summary>
        /// Raycast the sdf group. This is done via raymarching on the CPU side.
        /// </summary>
        public bool Raycast(Vector3 origin, Vector3 direction, out Vector3 hitPoint, out Vector3 hitNormal, float maxDist = 350f) =>
            Mapper.Raymarch(origin, direction, out hitPoint, out hitNormal, maxDist);

        #endregion

        #region Helper Methods

        private void ClearNulls<T>(List<T> list) where T : MonoBehaviour
        {
            for (int i = list.Count - 1; i >= 0; --i)
                if (!list[i])
                    list.RemoveAt(i);
        }

        #endregion
    }
}