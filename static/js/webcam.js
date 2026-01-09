class WebcamHandler {
    constructor(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.stream = null;
        this.isRunning = false;
    }
    
    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            this.isRunning = true;
            
            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    resolve(true);
                };
            });
        } catch (error) {
            console.error('Error accessing webcam:', error);
            throw error;
        }
    }
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.isRunning = false;
        }
    }
    
    capture(format = 'image/jpeg', quality = 0.9) {
        if (!this.isRunning) {
            throw new Error('Webcam is not running');
        }
        
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        const ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0);
        
        return this.canvas.toDataURL(format, quality);
    }
    
    captureMirrored(format = 'image/jpeg', quality = 0.9) {
        if (!this.isRunning) {
            throw new Error('Webcam is not running');
        }
        
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        const ctx = this.canvas.getContext('2d');
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(this.video, -this.canvas.width, 0);
        ctx.restore();
        
        return this.canvas.toDataURL(format, quality);
    }
}

window.WebcamHandler = WebcamHandler;
