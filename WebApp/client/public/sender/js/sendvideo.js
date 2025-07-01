export class SendVideo {
    constructor(localVideoElement) {
        this.localVideo = localVideoElement;
        this.peerConnection = null;
        this.videoTrack = null;
        this.framePathTemplate = "/videos/compressed_";
        this.frameExtension = ".jpeg";
        this.frameIndex = 0;
        this.totalFrames = 896;
        this.frameBuffer = [];
        this.bufferSize = 896;
        this.active = true;
        this.streamStopped = false;
        this.animationFrameId = null;
        this.framesSent = 0;
        this.bytesSent = 0;
        this.initializePeerConnection();
        this.startFrameSequenceStream();
        document.addEventListener("visibilitychange", () => {
            this.active = document.visibilityState === "visible";
        });
    }

    getLocalTracks() {
        return this.videoTrack ? [this.videoTrack] : [];
    }

    async preloadFrames() {
        const promises = [];
        for (let i = 0; i < this.bufferSize; i++) {
            const idx = (this.frameIndex + i) % this.totalFrames;
            const path = `${this.framePathTemplate}${String(idx).padStart(4, "0")}${this.frameExtension}`;
            promises.push(new Promise(resolve => {
                const img = new Image();
                img.crossOrigin = "anonymous";
                img.onload = () => resolve(img);
                img.onerror = () => resolve(null);
                img.src = path;
            }));
        }
        const results = await Promise.all(promises);
        this.frameBuffer.push(...results.filter(img => img));
        console.log("Frames preloaded:", this.frameBuffer.length);
    }

    startFrameSequenceStream() {
        (async () => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            const targetWidth = 1920;
            const targetHeight = 960;
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            await this.preloadFrames();
            const targetFPS = 1000;
            const targetSourceFPS = 1000; // set to your video FPS
            const frameDuration = 1000 / targetSourceFPS;
            const stream = canvas.captureStream(targetFPS);
            this.localVideo.srcObject = stream;
            this.videoTrack = stream.getVideoTracks()[0];
            this.videoTrack.contentHint = "detail";
            setTimeout(() => {
                if (this.peerConnection) {
                    const sender = this.peerConnection.addTrack(this.videoTrack, stream);
                    this.enforceBitrate(sender, 50000000);
                    this.disableResolutionScaling(sender);
                }
            }, 2000);
            let lastFrameTime = performance.now();
            const updateFrame = () => {
                if (!this.active || this.streamStopped) return;
                const now = performance.now();
                if (now - lastFrameTime >= frameDuration && this.frameBuffer.length > 0) {
                    const img = this.frameBuffer.shift();
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    this.framesSent++;
                    if (this.framesSent % 60 === 0) {
                        console.log(`[SENT] Frames sent: ${this.framesSent}`);
                    }
                    this.frameIndex = (this.frameIndex + 1) % this.totalFrames;
                    this.preloadNextFrame();
                    lastFrameTime = now;
                }
                this.animationFrameId = requestAnimationFrame(updateFrame);
            };
            this.animationFrameId = requestAnimationFrame(updateFrame);
        })().catch(e => console.error(e));
    }

    preloadNextFrame() {
        const idx = (this.frameIndex + this.bufferSize) % this.totalFrames;
        const path = `${this.framePathTemplate}${String(idx).padStart(4, "0")}${this.frameExtension}`;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => this.frameBuffer.push(img);
        img.onerror = () => {};
        img.src = path;
    }

    stopStream() {
        this.streamStopped = true;
        if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
        if (this.videoTrack) {
            this.videoTrack.stop();
        }
        if (this.localVideo) {
            this.localVideo.srcObject = null;
        }
        console.log("Stream stopped");
    }

    initializePeerConnection() {
        if (this.peerConnection) return;
        this.peerConnection = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
        this.peerConnection.onconnectionstatechange = () => {};
        this.peerConnection.onsignalingstatechange = () => {};
        this.peerConnection.oniceconnectionstatechange = () => {};
        this.createAndSendOffer();
        console.log("PeerConnection initialized");
    }

    enforceBitrate(sender, bitrate) {
        const params = sender.getParameters();
        if (!params.encodings) params.encodings = [{}];
        params.encodings[0].maxBitrate = bitrate;
        params.encodings[0].minBitrate = bitrate / 2;
        sender.setParameters(params).catch(e => {});
        console.log("Bitrate enforced:", bitrate);
    }

    disableResolutionScaling(sender) {
        const params = sender.getParameters();
        if (!params.encodings) params.encodings = [{}];
        params.encodings[0].scaleResolutionDownBy = 1;
        sender.setParameters(params).catch(e => {});
    }

    createAndSendOffer() {
        this.peerConnection
            .createOffer()
            .then(offer => this.peerConnection.setLocalDescription(offer))
            .catch(e => {});
        console.log("Offer created and set as local description");
    }
}
