# AcoustiX — Dual-Microphone ANC / Speech Enhancement

The implementation lives in [anc-system](anc-system). It contains a
two-microphone, low-latency speech-enhancement prototype: NLMS adaptive noise
cancellation, Wiener suppression, and a neural refinement stage.

Trained checkpoints are included so a collaborator can inspect, export, and
run the models without the local training audio. Audio corpora and generated
manifests are deliberately excluded from the repository.

See [anc-system/README.md](anc-system/README.md) for setup, checkpoints,
evaluation results, ONNX export, and microphone testing.
