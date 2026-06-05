import cv2
import numpy as np


class HeatmapVisualizer:
    FIELD_WIDTH    = 68.0   # metres  (position_transformed[1])
    FIELD_DEPTH    = 23.32  # metres  (position_transformed[0])
    SUMMARY_FRAMES = 72     # ~3 s at 24 fps

    def _collect_positions(self, tracks, team_id):
        positions = []
        for player_track in tracks['players']:
            for _, info in player_track.items():
                if info.get('team') != team_id:
                    continue
                pos = info.get('position_transformed')
                if pos is not None:
                    positions.append(pos)
        return positions

    def _build_heatmap(self, positions, canvas_w, canvas_h):
        density = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        for fx, fy in positions:
            rx = int(fy / self.FIELD_WIDTH  * canvas_w)
            ry = int(fx / self.FIELD_DEPTH  * canvas_h)
            if 0 <= rx < canvas_w and 0 <= ry < canvas_h:
                density[ry, rx] += 1

        cv2.GaussianBlur(density, (31, 31), 0, density)
        if density.max() > 0:
            cv2.normalize(density, density, 0, 255, cv2.NORM_MINMAX)

        colored = cv2.applyColorMap(density.astype(np.uint8), cv2.COLORMAP_JET)

        # Pitch markings
        cv2.rectangle(colored, (2, 2), (canvas_w - 3, canvas_h - 3), (255, 255, 255), 2)
        cv2.line(colored, (canvas_w // 2, 2), (canvas_w // 2, canvas_h - 3), (255, 255, 255), 1)
        cv2.circle(colored, (canvas_w // 2, canvas_h // 2), 4, (255, 255, 255), -1)

        return colored

    def _draw_legend(self, summary, x, y, width, height=14):
        """Horizontal gradient bar: blue (cold) → red (hot)."""
        bar = np.zeros((height, width, 1), dtype=np.uint8)
        for i in range(width):
            bar[:, i, 0] = int(i / width * 255)
        bar_colored = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
        summary[y:y + height, x:x + width] = bar_colored
        cv2.putText(summary, "cold", (x, y + height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        cv2.putText(summary, "hot", (x + width - 30, y + height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

    def generate_summary_frames(self, frames, tracks):
        if not frames:
            return frames

        fh, fw = frames[0].shape[:2]

        # Each heatmap takes ~45 % of frame width; height is proportional to field
        hm_w = int(fw * 0.42)
        hm_h = max(int(hm_w * self.FIELD_DEPTH / self.FIELD_WIDTH), 140)

        hm1 = self._build_heatmap(self._collect_positions(tracks, 1), hm_w, hm_h)
        hm2 = self._build_heatmap(self._collect_positions(tracks, 2), hm_w, hm_h)

        summary = np.full((fh, fw, 3), (18, 18, 18), dtype=np.uint8)

        # ── Title ─────────────────────────────────────────────────────────────
        title = "Player Heatmaps"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        cv2.putText(summary, title, ((fw - tw) // 2, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

        # ── Heatmaps ──────────────────────────────────────────────────────────
        y0 = (fh - hm_h) // 2 - 10

        for label, heatmap, col_center in [
            ("Team 1", hm1, fw // 4),
            ("Team 2", hm2, 3 * fw // 4),
        ]:
            x0 = col_center - hm_w // 2

            # Team label
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
            cv2.putText(summary, label, (col_center - lw // 2, y0 - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (210, 210, 210), 2)

            # Paste heatmap
            summary[y0:y0 + hm_h, x0:x0 + hm_w] = heatmap
            cv2.rectangle(summary,
                          (x0 - 3, y0 - 3), (x0 + hm_w + 3, y0 + hm_h + 3),
                          (255, 255, 255), 2)

            # Legend under each heatmap
            leg_w = hm_w // 2
            self._draw_legend(summary, col_center - leg_w // 2, y0 + hm_h + 14, leg_w)

        return frames + [summary] * self.SUMMARY_FRAMES
