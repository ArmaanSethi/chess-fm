/**
 * Layer B Optional Local WebGPU AI Coach Contract (``PROJECT.md`` Interface Contract #2).
 *
 * Dynamically imported by ``web/js/app.js`` ONLY when the user opts in via
 * "Enable Local AI Coach (WebGPU)". In Layer A (before Milestone M5 uploads the
 * distilled ONNX WebGPU student weights to Hugging Face ``winsonly``), this module
 * cleanly surfaces capability and readiness status and triggers seamless Layer A
 * deterministic fallback without throwing uncaught errors.
 */

/**
 * Check whether the current browser runtime exposes WebGPU (``navigator.gpu``).
 *
 * @returns {boolean}
 */
export function isWebGpuSupported() {
  return typeof navigator !== 'undefined' && Boolean(navigator.gpu);
}

/**
 * Optional local WebGPU coach client adhering to Interface Contract #2.
 */
export class WebGPUCoach {
  /**
   * @param {{ modelId?: string | null, adapterCheck?: () => Promise<boolean> }} [options]
   */
  constructor(options = {}) {
    this.modelId = options.modelId ?? null;
    this.adapterCheck = options.adapterCheck || null;
    this.status = 'idle'; // 'idle' | 'unsupported' | 'pending_weights' | 'loading' | 'ready' | 'error'
    this.statusMessage = 'Layer B WebGPU coach is off (using instant Layer A verified coach).';
    this.session = null;
  }

  /**
   * Probe WebGPU support and initialize the local ONNX WebGPU model if configured.
   *
   * @param {{ onProgress?: (info: { phase: string, pct: number, message: string }) => void }} [options]
   * @returns {Promise<{ ready: boolean, status: string, message: string }>}
   */
  async init(options = {}) {
    const onProgress = options.onProgress || (() => {});

    if (!isWebGpuSupported()) {
      this.status = 'unsupported';
      this.statusMessage =
        'WebGPU is unavailable in this browser — using verified Layer A deterministic coach.';
      onProgress({ phase: 'unsupported', pct: 0, message: this.statusMessage });
      return { ready: false, status: this.status, message: this.statusMessage };
    }

    if (this.adapterCheck) {
      const hasAdapter = await this.adapterCheck();
      if (!hasAdapter) {
        this.status = 'unsupported';
        this.statusMessage =
          'No WebGPU GPU adapter found — using verified Layer A deterministic coach.';
        onProgress({ phase: 'unsupported', pct: 0, message: this.statusMessage });
        return { ready: false, status: this.status, message: this.statusMessage };
      }
    }

    if (!this.modelId) {
      this.status = 'pending_weights';
      this.statusMessage =
        'Layer B WebGPU weights pending M5 upload — using verified Layer A deterministic coach.';
      onProgress({ phase: 'pending_weights', pct: 0, message: this.statusMessage });
      return { ready: false, status: this.status, message: this.statusMessage };
    }

    // M5 will wire Transformers.js / ONNX WebGPU inference here for `winsonly/<modelId>`.
    this.status = 'pending_weights';
    this.statusMessage =
      `Model '${this.modelId}' not yet loaded in Layer A — using verified Layer A deterministic coach.`;
    onProgress({ phase: 'pending_weights', pct: 0, message: this.statusMessage });
    return { ready: false, status: this.status, message: this.statusMessage };
  }

  /**
   * Generate a coaching explanation conforming to Interface Contract #2:
   * returns ``{ text, latencyMs, modelId, fallback: boolean, reason?: string }``.
   *
   * @param {{ fen: string, taskType: string, context?: any, onProgress?: Function }} params
   */
  async generate({ fen, taskType, context = {}, onProgress = null }) {
    const start = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (this.status !== 'ready' || !this.session) {
      if (onProgress) {
        onProgress({ phase: 'fallback', pct: 100, message: this.statusMessage });
      }
      const end = typeof performance !== 'undefined' ? performance.now() : Date.now();
      return {
        text: null,
        latencyMs: Math.max(0, Math.round(end - start)),
        modelId: this.modelId,
        fallback: true,
        reason: this.statusMessage,
        fen,
        taskType,
        context,
      };
    }

    // Placeholder path for M5 once `this.session` is wired
    const end = typeof performance !== 'undefined' ? performance.now() : Date.now();
    return {
      text: null,
      latencyMs: Math.max(0, Math.round(end - start)),
      modelId: this.modelId,
      fallback: true,
      reason: 'Layer B session inactive — falling back to Layer A deterministic coach.',
    };
  }
}
