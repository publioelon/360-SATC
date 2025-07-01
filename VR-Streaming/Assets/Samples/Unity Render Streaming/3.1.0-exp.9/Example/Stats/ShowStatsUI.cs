using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using System.Text;
using Unity.WebRTC;
using UnityEngine;
using UnityEngine.UI;

namespace Unity.RenderStreaming.Samples
{
    internal class ShowStatsUI : MonoBehaviour
    {
        [SerializeField] private Button showStatsButton;
        [SerializeField] private Button hideStatsButton;
        [SerializeField] private GameObject scrollView;
        [SerializeField] private RectTransform displayParent;
        [SerializeField] private Text baseText;
        [SerializeField] private List<SignalingHandlerBase> signalingHandlerList;

        private Dictionary<string, HashSet<RTCRtpSender>> activeSenderList =
            new Dictionary<string, HashSet<RTCRtpSender>>();

        private Dictionary<StreamReceiverBase, HashSet<RTCRtpReceiver>> activeReceiverList =
            new Dictionary<StreamReceiverBase, HashSet<RTCRtpReceiver>>();

        private Dictionary<RTCRtpSender, StatsDisplay> lastSenderStats =
            new Dictionary<RTCRtpSender, StatsDisplay>();

        private Dictionary<RTCRtpReceiver, StatsDisplay> lastReceiverStats =
            new Dictionary<RTCRtpReceiver, StatsDisplay>();

        private HashSet<StreamSenderBase> alreadySetupSenderList = new HashSet<StreamSenderBase>();

        private void Awake()
        {
            showStatsButton.onClick.AddListener(ShowStats);
            hideStatsButton.onClick.AddListener(HideStats);
        }

        private void ShowStats()
        {
            scrollView.gameObject.SetActive(true);
            hideStatsButton.gameObject.SetActive(true);
            showStatsButton.gameObject.SetActive(false);
        }

        private void HideStats()
        {
            scrollView.gameObject.SetActive(false);
            showStatsButton.gameObject.SetActive(true);
            hideStatsButton.gameObject.SetActive(false);
        }

        private void Start()
        {
            StartCoroutine(CollectStats());
        }

        private void OnDestroy()
        {
            lastSenderStats.Clear();
            lastReceiverStats.Clear();
            activeSenderList.Clear();
            activeReceiverList.Clear();
            alreadySetupSenderList.Clear();
        }

        public void AddSignalingHandler(SignalingHandlerBase handlerBase)
        {
            if (signalingHandlerList.Contains(handlerBase))
            {
                return;
            }

            signalingHandlerList.Add(handlerBase);
        }

        class StatsDisplay
        {
            public Text display;
            public RTCStatsReport lastReport;
        }

        private IEnumerator CollectStats()
        {
            var waitSec = new WaitForSeconds(1);

            while (true)
            {
                yield return waitSec;

                foreach (var streamBase in signalingHandlerList.SelectMany(x => x.Streams))
                {
                    if (streamBase is StreamSenderBase senderBase)
                    {
                        SetUpSenderBase(senderBase);
                    }

                    if (streamBase is StreamReceiverBase receiverBase)
                    {
                        SetUpReceiverBase(receiverBase);
                    }
                }

                List<Coroutine> coroutines = new List<Coroutine>();

                foreach (var sender in activeSenderList.Values.SelectMany(x => x))
                {
                    var coroutine = StartCoroutine(UpdateStats(sender));
                    coroutines.Add(coroutine);
                }

                foreach (var receiver in activeReceiverList.Values.SelectMany(x => x))
                {
                    var coroutine = StartCoroutine(UpdateStats(receiver));
                    coroutines.Add(coroutine);
                }
                foreach (var coroutine in coroutines)
                {
                    yield return coroutine;
                }
                var noStatsData = !lastSenderStats.Any() && !lastReceiverStats.Any();
                baseText.gameObject.SetActive(noStatsData);
            }
        }

        IEnumerator UpdateStats(RTCRtpReceiver receiver)
        {
            var op = receiver.GetStats();
            yield return new WaitUntilWithTimeout(() => op.IsDone, 3f);

            if (op.IsError || !op.IsDone)
            {
                yield break;
            }

            var report = op.Value;
            if (report == null)
            {
                yield break;
            }

            if (lastReceiverStats.TryGetValue(receiver, out var statsDisplay))
            {
                var lastReport = statsDisplay.lastReport;
                statsDisplay.display.text = CreateDisplayString(report, lastReport);
                statsDisplay.lastReport = report;
                lastReport.Dispose();
            }
            else
            {
                var text = Instantiate(baseText, displayParent);
                text.text = "";
                text.gameObject.SetActive(true);
                lastReceiverStats[receiver] = new StatsDisplay { display = text, lastReport = report };
            }
        }

        IEnumerator UpdateStats(RTCRtpSender sender)
        {
            var op = sender.GetStats();
            yield return new WaitUntilWithTimeout(() => op.IsDone, 3f);

            if (op.IsError || !op.IsDone)
            {
                yield break;
            }

            var report = op.Value;
            if (report == null)
            {
                yield break;
            }

            if (lastSenderStats.TryGetValue(sender, out var statsDisplay))
            {
                var lastReport = statsDisplay.lastReport;
                statsDisplay.display.text = CreateDisplayString(report, lastReport);
                statsDisplay.lastReport = report;
                lastReport.Dispose();
            }
            else
            {
                var text = Instantiate(baseText, displayParent);
                text.text = "";
                text.gameObject.SetActive(true);
                lastSenderStats[sender] = new StatsDisplay { display = text, lastReport = report };
            }
        }

        private void SetUpSenderBase(StreamSenderBase senderBase)
        {
            if (alreadySetupSenderList.Contains(senderBase))
            {
                return;
            }

            senderBase.OnStartedStream += id =>
            {
                if (!activeSenderList.ContainsKey(id))
                {
                    activeSenderList[id] = new HashSet<RTCRtpSender>();
                }

                if (senderBase.Transceivers.TryGetValue(id, out var transceiver))
                {
                    activeSenderList[id].Add(transceiver.Sender);
                }
            };
            senderBase.OnStoppedStream += id =>
            {
                if (activeSenderList.TryGetValue(id, out var hashSet))
                {
                    foreach (var sender in hashSet)
                    {
                        if (lastSenderStats.TryGetValue(sender, out var statsDisplay))
                        {
                            DestroyImmediate(statsDisplay.display.gameObject);
                            lastSenderStats.Remove(sender);
                        }
                    }
                }
                activeSenderList.Remove(id);
            };

            foreach (var pair in senderBase.Transceivers)
            {
                if (!activeSenderList.ContainsKey(pair.Key))
                {
                    activeSenderList[pair.Key] = new HashSet<RTCRtpSender>();
                }

                activeSenderList[pair.Key].Add(pair.Value.Sender);
            }

            alreadySetupSenderList.Add(senderBase);
        }

        private void SetUpReceiverBase(StreamReceiverBase receiverBase)
        {
            if (activeReceiverList.ContainsKey(receiverBase))
            {
                return;
            }

            activeReceiverList[receiverBase] = new HashSet<RTCRtpReceiver>();

            receiverBase.OnStartedStream += id =>
            {
                if (activeReceiverList.TryGetValue(receiverBase, out var hashSet))
                {
                    hashSet.Add(receiverBase.Transceiver.Receiver);
                }
            };
            receiverBase.OnStoppedStream += id =>
            {
                if (activeReceiverList.TryGetValue(receiverBase, out var hashSet))
                {
                    foreach (var receiver in hashSet)
                    {
                        if (lastReceiverStats.TryGetValue(receiver, out var statsDisplay))
                        {
                            DestroyImmediate(statsDisplay.display.gameObject);
                            lastReceiverStats.Remove(receiver);
                        }
                    }
                }
                activeReceiverList.Remove(receiverBase);
            };

            var transceiver = receiverBase.Transceiver;
            if (transceiver != null && transceiver.Receiver != null)
            {
                activeReceiverList[receiverBase].Add(transceiver.Receiver);
            }
        }
    private static readonly Dictionary<string, List<double>> fpsHistory = new Dictionary<string, List<double>>();
    private static readonly Dictionary<string, ulong> frameCountHistory = new Dictionary<string, ulong>();
    static Dictionary<string, ulong> cumulativeBytesReceived = new Dictionary<string, ulong>();

    public static string CreateDisplayString(RTCStatsReport report, RTCStatsReport lastReport)
    {
        var builder = new StringBuilder();

        // You can sum across all inbound video stream IDs at the end:
        ulong grandTotalVideoBytes = 0;

        foreach (var stats in report.Stats.Values)
        {
            if (stats is RTCInboundRTPStreamStats inboundStats)
            {
                builder.AppendLine($"{inboundStats.kind} receiving stream stats");
                // Codec details
                if (inboundStats.codecId != null && report.Get(inboundStats.codecId) is RTCCodecStats codecStats)
                {
                    builder.AppendLine($"Codec: {codecStats.mimeType}");
                    if (!string.IsNullOrEmpty(codecStats.sdpFmtpLine))
                        foreach (var fmtp in codecStats.sdpFmtpLine.Split(';'))
                            builder.AppendLine($" - {fmtp}");
                    if (codecStats.payloadType > 0)
                        builder.AppendLine($" - payloadType={codecStats.payloadType}");
                    if (codecStats.clockRate > 0)
                        builder.AppendLine($" - clockRate={codecStats.clockRate}");
                    if (codecStats.channels > 0)
                        builder.AppendLine($" - channels={codecStats.channels}");
                }

                // --- BYTE COUNTING ---
                // Track cumulative received bytes for each inbound RTP stream (by stream Id)
                ulong currentBytes = inboundStats.bytesReceived;
                cumulativeBytesReceived[inboundStats.Id] = currentBytes;

                if (inboundStats.kind == "video")
                {
                    builder.AppendLine($"[VIDEO] Cumulative Bytes Received: {currentBytes} bytes ({(currentBytes/1024.0/1024.0):F2} MB)");
                    grandTotalVideoBytes += currentBytes;
                }

                // Decoder/resolution/instant-fps
                if (inboundStats.kind == "video")
                {
                    builder.AppendLine($"Decoder: {inboundStats.decoderImplementation}");
                    builder.AppendLine($"Resolution: {inboundStats.frameWidth}x{inboundStats.frameHeight}");
                    builder.AppendLine($"Instant FPS: {inboundStats.framesPerSecond:F2}");

                    // Packet loss
                    builder.AppendLine($"Packets Received: {inboundStats.packetsReceived}");
                    builder.AppendLine($"Packets Lost:     {inboundStats.packetsLost}");
                    var pktTotal = inboundStats.packetsReceived + inboundStats.packetsLost;
                    if (pktTotal > 0)
                        builder.AppendLine($"Packet Loss %:    {((double)inboundStats.packetsLost / pktTotal * 100):F2}%");

                    var id = inboundStats.Id;
                    ulong totalFrames = inboundStats.framesReceived;
                    frameCountHistory.TryGetValue(id, out var prevCount);
                    ulong deltaFrames = totalFrames - prevCount;
                    frameCountHistory[id] = totalFrames;

                    builder.AppendLine($"Total frames received: {totalFrames}");
                    builder.AppendLine($"New frames since last check: {deltaFrames}");

                    // Frame drop
                    builder.AppendLine($"Frames Decoded:    {inboundStats.framesDecoded}");
                    builder.AppendLine($"Frames Dropped:    {inboundStats.framesDropped}");
                    var frmTotal = inboundStats.framesDecoded + inboundStats.framesDropped;
                    if (frmTotal > 0)
                        builder.AppendLine($"Frame Drop %:      {((double)inboundStats.framesDropped / frmTotal * 100):F2}%");

                    if (!fpsHistory.TryGetValue(id, out var list))
                    {
                        list = new List<double>();
                        fpsHistory[id] = list;
                    }
                    list.Add(inboundStats.framesPerSecond);

                    double avg = list.Average();
                    double variance = list.Average(v => (v - avg) * (v - avg));
                    double stddev = Math.Sqrt(variance);

                    var sorted = list.OrderBy(v => v).ToList();
                    double median;
                    int n = sorted.Count;
                    if (n % 2 == 1)
                        median = sorted[n / 2];
                    else
                        median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;

                    double minFps = sorted.First();
                    double maxFps = sorted.Last();

                    builder.AppendLine($"Average FPS: {avg:F2}");
                    builder.AppendLine($"FPS StdDev: {stddev:F2}");
                    builder.AppendLine($"Median FPS: {median:F2}");
                    builder.AppendLine($"Lowest FPS: {minFps:F2}");
                    builder.AppendLine($"Highest FPS: {maxFps:F2}");
                }

                if (lastReport.TryGetValue(inboundStats.Id, out var lastStats) &&
                    lastStats is RTCInboundRTPStreamStats lastInboundStats)
                {
                    var duration = (double)(inboundStats.Timestamp - lastInboundStats.Timestamp) / 1_000_000;
                    var bitrate = (8 * (inboundStats.bytesReceived - lastInboundStats.bytesReceived) / duration) / 1000;
                    builder.AppendLine($"Bitrate: {bitrate:F2} kbit/sec");
                }
            }
            else if (stats is RTCOutboundRTPStreamStats outboundStats)
            {
                builder.AppendLine($"{outboundStats.kind} sending stream stats");
                // Codec details
                if (outboundStats.codecId != null && report.Get(outboundStats.codecId) is RTCCodecStats codecStats)
                {
                    builder.AppendLine($"Codec: {codecStats.mimeType}");
                    if (!string.IsNullOrEmpty(codecStats.sdpFmtpLine))
                        foreach (var fmtp in codecStats.sdpFmtpLine.Split(';'))
                            builder.AppendLine($" - {fmtp}");
                    if (codecStats.payloadType > 0)
                        builder.AppendLine($" - payloadType={codecStats.payloadType}");
                    if (codecStats.clockRate > 0)
                        builder.AppendLine($" - clockRate={codecStats.clockRate}");
                    if (codecStats.channels > 0)
                        builder.AppendLine($" - channels={codecStats.channels}");
                }

                if (outboundStats.kind == "video")
                {
                    // Encoder/resolution/instant-fps
                    builder.AppendLine($"Encoder: {outboundStats.encoderImplementation}");
                    builder.AppendLine($"Resolution: {outboundStats.frameWidth}x{outboundStats.frameHeight}");
                    builder.AppendLine($"Instant FPS: {outboundStats.framesPerSecond:F2}");

                    // Outbound FPS statistics
                    var id = outboundStats.Id;
                    if (!fpsHistory.TryGetValue(id, out var list))
                    {
                        list = new List<double>();
                        fpsHistory[id] = list;
                    }
                    list.Add(outboundStats.framesPerSecond);

                    double avg = list.Average();
                    double variance = list.Average(v => (v - avg) * (v - avg));
                    double stddev = Math.Sqrt(variance);

                    var sorted = list.OrderBy(v => v).ToList();
                    double median;
                    int n = sorted.Count;
                    if (n % 2 == 1)
                        median = sorted[n / 2];
                    else
                        median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;

                    double minFps = sorted.First();
                    double maxFps = sorted.Last();

                    builder.AppendLine($"Average FPS: {avg:F2}");
                    builder.AppendLine($"FPS StdDev: {stddev:F2}");
                    builder.AppendLine($"Median FPS: {median:F2}");
                    builder.AppendLine($"Lowest FPS: {minFps:F2}");
                    builder.AppendLine($"Highest FPS: {maxFps:F2}");
                }

                // Calculate outbound bitrate
                if (lastReport.TryGetValue(outboundStats.Id, out var lastStats) &&
                    lastStats is RTCOutboundRTPStreamStats lastOutboundStats)
                {
                    var duration = (double)(outboundStats.Timestamp - lastOutboundStats.Timestamp) / 1_000_000;
                    var bitrate = (8 * (outboundStats.bytesSent - lastOutboundStats.bytesSent) / duration) / 1000;
                    builder.AppendLine($"Bitrate: {bitrate:F2} kbit/sec");
                }
            }
        }

        // At the end, print GRAND TOTAL for all video streams
        if (grandTotalVideoBytes > 0)
            builder.AppendLine($"\n[VIDEO] GRAND TOTAL Bytes Received (all streams): {grandTotalVideoBytes} bytes ({(grandTotalVideoBytes / 1024.0 / 1024.0):F2} MB)");

        return builder.ToString();
    }
}

    internal class WaitUntilWithTimeout : CustomYieldInstruction
    {
        public bool IsCompleted { get; private set; }

        private readonly float timeoutTime;

        private readonly System.Func<bool> predicate;

        public override bool keepWaiting
        {
            get
            {
                IsCompleted = predicate();
                if (IsCompleted)
                {
                    return false;
                }

                return !(Time.realtimeSinceStartup >= timeoutTime);
            }
        }

        public WaitUntilWithTimeout(System.Func<bool> predicate, float timeout)
        {
            this.timeoutTime = Time.realtimeSinceStartup + timeout;
            this.predicate = predicate;
        }
    }
}
