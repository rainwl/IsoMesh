using System;
using IsoMesh;
using UnityEditor;
using UnityEngine;
using UnityEngine.Serialization;

namespace Source.SDFs
{
    public class SDFBone : MonoBehaviour
    {
        public SDFMeshAsset boneSDF;

        private void OnEnable()
        {
            //boneSDF = Resources.Load<SDFMeshAsset>($"Assets/Data/SDFMeshes/SDFMesh_SM_YZ_L4_G_64");

            Debug.Log($"xixiix"+boneSDF.MaxBounds);
        }
    }
}