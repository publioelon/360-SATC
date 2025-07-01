import { SendVideo } from "./sendvideo.js";
import { getServerConfig, getRTCConfiguration } from "../../js/config.js";
import { RenderStreaming } from "../../module/renderstreaming.js";
import { Signaling, WebSocketSignaling } from "../../module/signaling.js";

const localVideo = document.getElementById("localVideo");
const textForConnectionId = document.getElementById("textForConnectionId");
textForConnectionId.value = getRandom();
const codecPreferences = document.getElementById("codecPreferences");

let sendVideo;
let renderstreaming;
let useWebSocket;
let connectionId;

const startButton = document.getElementById("startVideoButton");
startButton.addEventListener("click", startVideo);
const setupButton = document.getElementById("setUpButton");
setupButton.addEventListener("click", setUp);
const hangUpButton = document.getElementById("hangUpButton");
hangUpButton.addEventListener("click", hangUp);

window.addEventListener(
  "beforeunload",
  async () => {
    if (!renderstreaming) return;
    await renderstreaming.stop();
  },
  true
);

setupConfig();

async function setupConfig() {
  const res = await getServerConfig();
  useWebSocket = res.useWebSocket;
  const warningDiv = document.getElementById("warning");
  if (res.startupMode === "public") {
    warningDiv.innerHTML = "<h4>Warning</h4> This sample is not working on Public Mode.";
    warningDiv.hidden = false;
  }
  const codecs = RTCRtpSender.getCapabilities("video").codecs;
  codecs.forEach((codec) => {
    if (["video/red", "video/ulpfec", "video/rtx"].includes(codec.mimeType)) return;
    const option = document.createElement("option");
    option.value = (codec.mimeType + " " + (codec.sdpFmtpLine || "")).trim();
    option.innerText = option.value;
    codecPreferences.appendChild(option);
  });
  const vp9Option = Array.from(codecPreferences.options).find((option) =>
    option.value.startsWith("video/vp9")
  );
  if (vp9Option) vp9Option.selected = true;
}

async function startVideo() {
  if (!sendVideo) {
    sendVideo = new SendVideo(localVideo, {
      video: {
        width: 4096,
        height: 2048,
        frameRate: 1000,
      },
    });
  }
  startButton.disabled = true;
  setupButton.disabled = false;
}

async function setUp() {
  setupButton.disabled = true;
  hangUpButton.disabled = false;
  connectionId = textForConnectionId.value;

  const signaling = useWebSocket ? new WebSocketSignaling() : new Signaling();
  const config = getRTCConfiguration();
  renderstreaming = new RenderStreaming(signaling, config);

  renderstreaming.onConnect = () => {
    const tracks = sendVideo.getLocalTracks();
    if (!tracks.length) {
      console.error("No local tracks available from SendVideo.");
      return;
    }
    tracks.forEach((track) => {
      const transceiver = renderstreaming.addTransceiver(track, {
        direction: "sendonly",
      });
      const sender = transceiver.sender;
      const parameters = sender.getParameters();
      if (!parameters.encodings) parameters.encodings = [{}];
      parameters.encodings[0].maxBitrate = 50000000;
      parameters.encodings[0].minBitrate = 25000000;
      parameters.encodings[0].scaleResolutionDownBy = 1;
      sender.setParameters(parameters).catch((e) => console.error(e));
    });
    const preferredCodec = codecPreferences.options[codecPreferences.selectedIndex];
    if (preferredCodec.value !== "") {
      const [mimeType, sdpFmtpLine] = preferredCodec.value.split(" ");
      const { codecs } = RTCRtpSender.getCapabilities("video");
      const selectedCodec = codecs.find(
        (c) => c.mimeType === mimeType && c.sdpFmtpLine === sdpFmtpLine
      );
      renderstreaming
        .getTransceivers()
        .filter((t) => t.receiver.track.kind === "video")
        .forEach((t) => t.setCodecPreferences([selectedCodec]));
    }
    console.log("Connected to the receiver.");
    const senders = renderstreaming.getTransceivers().map((t) => t.sender);
    setInterval(() => {
      senders.forEach((sender) => {
        sender.getStats().then((stats) => {
          stats.forEach((report) => {
            if (report.type === "outbound-rtp") {
              const sent = report.packetsSent || 0;
              const lost = report.packetsLost || 0;
              const loss = sent + lost ? ((lost / (sent + lost)) * 100).toFixed(2) : "0.00";
              const enc = report.framesEncoded || 0;
              const fsent = report.framesSent || 0;
              console.log(
                `pktSent=${sent}, pktLost=${lost}(${loss}%), encFrames=${enc}, sentFrames=${fsent}`
              );
            }
          });
        });
      });
    }, 1000);
  };

  renderstreaming.onDisconnect = async () => {
    console.log("Disconnected from the receiver.");
    await hangUp();
  };

  await renderstreaming.start();
  await renderstreaming.createConnection(connectionId);
}

async function hangUp() {
  console.log("Disconnecting...");
  if (renderstreaming) {
    await renderstreaming.deleteConnection();
    await renderstreaming.stop();
    renderstreaming = null;
  }
  connectionId = null;
  textForConnectionId.value = getRandom();
}

function getRandom() {
  const max = 99999;
  const length = String(max).length;
  const number = Math.floor(Math.random() * max);
  return (Array(length).join("0") + number).slice(-length);
}
