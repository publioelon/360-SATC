using System;
using UnityEngine;
using UnityEngine.UI;

namespace Unity.RenderStreaming.Samples
{
    static class InputSenderExtension
    {
        public static (Rect, Vector2Int) GetRegionAndSize(this RawImage image)
        {
            Vector3[] corners = new Vector3[4];
            image.rectTransform.GetWorldCorners(corners);
            Camera camera = image.canvas.worldCamera;
            var corner0 = RectTransformUtility.WorldToScreenPoint(camera, corners[0]);
            var corner2 = RectTransformUtility.WorldToScreenPoint(camera, corners[2]);
            var region = new Rect(
                corner0.x,
                corner0.y,
                corner2.x - corner0.x,
                corner2.y - corner0.y
            );

            var size = new Vector2Int(image.texture.width, image.texture.height);
            return (region, size);
        }
    }

    class ReceiverSample : MonoBehaviour
    {
#pragma warning disable 0649
        [SerializeField] private SignalingManager renderStreaming;
        [SerializeField] private Button startButton;
        [SerializeField] private Button stopButton;
        [SerializeField] private InputField connectionIdInput;
        [SerializeField] private RawImage remoteVideoImage; 
        [SerializeField] private AudioSource remoteAudioSource;
        [SerializeField] private VideoStreamReceiver receiveVideoViewer;
        [SerializeField] private AudioStreamReceiver receiveAudioViewer;
        [SerializeField] private SingleConnection connection;
        [SerializeField] private Text resolution;
        [SerializeField] private Material skyboxMaterial; 
        [SerializeField] private Canvas uiCanvas; 
        [SerializeField] private Camera mainCamera;
        [SerializeField] private Text frameCounterText; // Optional UI display for frame count
#pragma warning restore 0649

        private string connectionId;
        private InputSender inputSender;
        private RenderStreamingSettings settings;
        private Vector2 lastSize;
        private int receivedFrameCount = 0;
        private int totalFramesExpected = 896;

        void Awake()
        {
            startButton.onClick.AddListener(OnStart);
            stopButton.onClick.AddListener(OnStop);
            if (connectionIdInput != null)
                connectionIdInput.onValueChanged.AddListener(input => connectionId = input);

            receiveVideoViewer.OnUpdateReceiveTexture += OnUpdateReceiveTexture;
            receiveAudioViewer.OnUpdateReceiveAudioSource += source =>
            {
                source.loop = true;
                source.Play();
            };

            inputSender = GetComponent<InputSender>();
            inputSender.OnStartedChannel += OnStartedChannel;

            settings = SampleManager.Instance.Settings;
        }

        void Start()
        {
            if (skyboxMaterial == null)
            {
                skyboxMaterial = Resources.Load<Material>("SkyboxMaterial");
                if (skyboxMaterial == null)
                {
                    Debug.LogError("Skybox material is not assigned and could not be found in Resources!!");
                }
            }

            if (renderStreaming.runOnAwake)
                return;

            if (settings != null)
                renderStreaming.useDefaultSettings = settings.UseDefaultSettings;
            if (settings?.SignalingSettings != null)
                renderStreaming.SetSignalingSettings(settings.SignalingSettings);
            renderStreaming.Run();
        }

        private void Update()
        {
            if (remoteVideoImage != null)
            {
                var size = remoteVideoImage.rectTransform.sizeDelta;
                if (lastSize == size)
                    return;
                lastSize = size;
                CalculateInputRegion();
            }
        }

        void OnUpdateReceiveTexture(Texture texture)
        {
            if (texture != null)
            {
                receivedFrameCount++;
                Debug.Log($"[RECEIVER] Received Texture: {texture.width}x{texture.height} | Frame Count: {receivedFrameCount}");

                // UI update (optional)
                if (frameCounterText != null)
                {
                    frameCounterText.text = $"Frames Received: {receivedFrameCount}";
                }

                // Print when we reach 896
                if (receivedFrameCount == totalFramesExpected)
                {
                    Debug.LogWarning($"[RECEIVER] Reached expected total of {totalFramesExpected} frames!");
                    // Optionally: trigger events, send message, etc.
                }
            }
            else
            {
                Debug.LogWarning("Received a null texture!");
            }

            if (skyboxMaterial != null)
            {
                skyboxMaterial.mainTexture = texture;
                RenderSettings.skybox = skyboxMaterial;
            }
            else
            {
                Debug.LogWarning("Skybox material is not assigned!!");
            }
        }

        void OnStartedChannel(string connectionId)
        {
            CalculateInputRegion();
        }

        private void OnRectTransformDimensionsChange()
        {
            CalculateInputRegion();
        }

        void CalculateInputRegion()
        {
            if (inputSender == null || !inputSender.IsConnected || 
                (remoteVideoImage != null && remoteVideoImage.texture == null))
                return;

            if (remoteVideoImage != null)
            {
                var (region, size) = remoteVideoImage.GetRegionAndSize();
                resolution.text = $"{(int)region.width} x {(int)region.height}";
                inputSender.CalculateInputResion(region, size);
                inputSender.EnableInputPositionCorrection(true);
            }
        }

        private void OnStart()
        {
            // Reset frame counting
            receivedFrameCount = 0;
            if (frameCounterText != null)
                frameCounterText.text = "Frames Received: 0";

            if (string.IsNullOrEmpty(connectionId))
            {
                connectionId = Guid.NewGuid().ToString("N");
                connectionIdInput.text = connectionId;
            }
            connectionIdInput.interactable = false;
            if (settings != null)
                receiveVideoViewer.SetCodec(settings.ReceiverVideoCodec);
            receiveAudioViewer.targetAudioSource = remoteAudioSource;

            connection.CreateConnection(connectionId);
            startButton.gameObject.SetActive(false);
            stopButton.gameObject.SetActive(true);

            if (uiCanvas != null)
                uiCanvas.gameObject.SetActive(false);

            if (mainCamera != null)
            {
                mainCamera.transform.position = Vector3.zero; 
                mainCamera.fieldOfView = 90f; 
            }
        }

        private void OnStop()
        {
            connection.DeleteConnection(connectionId);
            connectionId = String.Empty;
            connectionIdInput.text = String.Empty;
            connectionIdInput.interactable = true;
            startButton.gameObject.SetActive(true);
            stopButton.gameObject.SetActive(false);
            
            if (uiCanvas != null)
                uiCanvas.gameObject.SetActive(true);

            // Reset the frame counter
            receivedFrameCount = 0;
            if (frameCounterText != null)
                frameCounterText.text = "Frames Received: 0";
        }
    }
}
