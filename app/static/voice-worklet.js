"use strict";
// Voice mode capture (AudioWorklet, runs on the audio thread): microphone audio at the device rate ->
// 16 kHz mono 16-bit PCM in 100 ms chunks for app/voice.py. Downsampling averages each output
// sample's input span, a simple low-pass that keeps speech clean for the recogniser.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;
    this.acc = 0;
    this.n = 0;
    this.out = new Int16Array(1600);
    this.len = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i];
      this.n += 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.n));
        this.out[this.len++] = v < 0 ? v * 32768 : v * 32767;
        this.acc = 0;
        this.n = 0;
        if (this.len === this.out.length) {
          this.port.postMessage(this.out.buffer.slice(0));
          this.len = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("sia-capture", Capture);
