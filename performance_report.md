
# Real-Time Performance Report

## Summary
- **Iterations**: 50
- **Video Resolution**: 320x240
- **Audio Chunk**: 1.0 second (44.1kHz)

## Results
### Video Pipeline (Image -> Features -> Predict)
- **Average Latency**: 67.76 ms
- **Min Latency**: 34.46 ms
- **Max Latency**: 310.48 ms
- **FPS Capacity**: 14.8 FPS

### Audio Pipeline (Waveform -> Features -> Predict)
- **Average Latency**: 786.08 ms
- **Min Latency**: 205.23 ms
- **Max Latency**: 23030.93 ms

## Benchmark
- **Target Video Latency**: < 200ms (Achieved: YES)
- **Target Audio Latency**: < 500ms (Achieved: NO)
    