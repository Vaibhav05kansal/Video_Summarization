def fuse_scores(visual_scores, audio_scores, alpha=0.5):
    fused_scores = [(alpha * v) + ((1 - alpha) * a) for v, a in zip(visual_scores, audio_scores)]
    return fused_scores
