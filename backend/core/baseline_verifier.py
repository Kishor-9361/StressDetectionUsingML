import numpy as np

class BaselineVerifier:
    """
    Validates calibration baseline sequences to ensure the user is in a calm,
    low-stress, and clean sensor state during calibration across diverse faces and vocal registers.
    """
    def __init__(self, calm_hr_threshold=100.0, face_min_confidence=0.55, voice_max_intensity=0.85):
        self.calm_hr_threshold = calm_hr_threshold
        self.face_min_confidence = face_min_confidence
        self.voice_max_intensity = voice_max_intensity

    def verify_face_baseline(self, features_chunk):
        """
        Validates a sequence of face features (shape [N, 18]).
        Checks if tracking was stable and user had a resting, calm expression.
        """
        if features_chunk is None or len(features_chunk) == 0:
            return {"status": "recalibration_needed", "reason": "No face landmark data received"}

        # confidence is at index 16
        conf_idx = 16 if features_chunk.shape[1] > 16 else 0
        confidences = features_chunk[:, conf_idx]
        mean_conf = np.mean(confidences)
        if mean_conf < self.face_min_confidence:
            return {
                "status": "low_confidence",
                "reason": f"Face tracking confidence slightly low: {mean_conf:.2f} (acceptable > {self.face_min_confidence})"
            }

        # head_tilt is at index 12
        if features_chunk.shape[1] > 12:
            movement = float(np.std(features_chunk[:, 12]))
            if movement > 0.35:
                return {
                    "status": "low_confidence",
                    "reason": f"Head movement variation: {movement:.3f}"
                }

        return {"status": "verified", "reason": "Stable face baseline verified across geometry"}

    def verify_voice_baseline(self, features_chunk):
        """
        Validates voice sequence (shape [N, 12]).
        Ensures the baseline speech is natural, non-shouted, and clean.
        """
        if features_chunk is None or len(features_chunk) == 0:
            return {"status": "verified", "reason": "No voice data (optional)"}

        # voice_intensity is at index 7
        intensities = features_chunk[:, 7] if features_chunk.shape[1] > 7 else [0.05]
        mean_intensity = float(np.mean(intensities))
        
        # Verify speech is in a calm, non-shouted range
        if mean_intensity > self.voice_max_intensity:
            return {
                "status": "recalibration_needed",
                "reason": f"Vocal intensity excessively high/shouting: {mean_intensity:.3f}"
            }

        # Verify pitch if available (index 0) is in human speaking range
        f0_vals = [f for f in features_chunk[:, 0] if f > 40]
        if f0_vals:
            mean_f0 = float(np.median(f0_vals))
            if mean_f0 < 50 or mean_f0 > 550:
                return {
                    "status": "low_confidence",
                    "reason": f"Vocal fundamental frequency outside typical range: {mean_f0:.1f} Hz"
                }

        return {"status": "verified", "reason": "Natural vocal baseline verified"}

    def verify_physio_baseline(self, features_chunk):
        """
        Validates physiological sequence (shape [N, 5]).
        Ensures the user's heart rate is in a normal resting state (not elevated).
        """
        if features_chunk is None or len(features_chunk) == 0:
            return {"status": "recalibration_needed", "reason": "No physiological data received"}

        # hr_mean is at index 0
        hrs = features_chunk[:, 0]
        mean_hr = np.mean(hrs)

        if mean_hr > self.calm_hr_threshold:
            return {
                "status": "recalibration_needed",
                "reason": f"Elevated heart rate detected: {mean_hr:.1f} bpm (expected calm resting HR < {self.calm_hr_threshold})"
            }

        return {"status": "verified", "reason": "Physiological baseline verified in resting range"}

    def verify_calibration_session(self, face_data=None, voice_data=None, physio_data=None):
        """
        Aggregates results for all modalities to grant or deny baseline verification status.
        """
        reports = {}
        if face_data is not None:
            reports["face"] = self.verify_face_baseline(face_data)
        if voice_data is not None:
            reports["voice"] = self.verify_voice_baseline(voice_data)
        if physio_data is not None:
            reports["physio"] = self.verify_physio_baseline(physio_data)

        # Determine overall status
        overall_status = "verified"
        reasons = []
        
        for mod, r in reports.items():
            if r["status"] == "recalibration_needed":
                overall_status = "recalibration_needed"
                reasons.append(f"{mod}: {r['reason']}")
            elif r["status"] == "low_confidence" and overall_status != "recalibration_needed":
                overall_status = "low_confidence"
                reasons.append(f"{mod}: {r['reason']}")

        return {
            "status": overall_status,
            "reports": reports,
            "summary_reason": "; ".join(reasons) if reasons else "Calibration verified successfully"
        }
