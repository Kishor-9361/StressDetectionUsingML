
# Real-Time Performance Report

## Summary
- **Iterations**: 50
- **Video Resolution**: 320x240
- **Audio Chunk**: 1.0 second (44.1kHz)

## Results
### Video Pipeline (Image -> Features -> Predict)
- **Average Latency**: 76.66 ms
- **Min Latency**: 55.19 ms
- **Max Latency**: 466.68 ms
- **FPS Capacity**: 13.0 FPS

### Audio Pipeline (Waveform -> Features -> Predict)
- **Average Latency**: 251.02 ms
- **Min Latency**: 55.34 ms
- **Max Latency**: 7343.59 ms

## Benchmark
- **Target Video Latency**: < 200ms (Achieved: YES)
- **Target Audio Latency**: < 500ms (Achieved: YES)
    