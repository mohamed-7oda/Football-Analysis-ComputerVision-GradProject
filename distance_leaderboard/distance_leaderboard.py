import cv2
import numpy as np


class DistanceLeaderboard:
    SUMMARY_FRAMES = 72  # ~3 s at 24 fps

    # Medal colours (gold, silver, bronze, rest)
    _RANK_COLORS = [
        (0, 215, 255),   # gold
        (192, 192, 192), # silver
        (0, 140, 255),   # bronze
    ]
    _DEFAULT_COLOR = (200, 200, 200)

    def _get_distances(self, tracks):
        """Return {player_id: {'team': t, 'distance': d}} using max cumulative distance."""
        player_data = {}
        for player_track in tracks['players']:
            for pid, info in player_track.items():
                dist = info.get('distance', 0)
                team = info.get('team')
                if pid not in player_data:
                    player_data[pid] = {'team': team, 'distance': 0.0}
                if dist > player_data[pid]['distance']:
                    player_data[pid]['distance'] = dist
                if team is not None:
                    player_data[pid]['team'] = team
        return player_data

    def generate_summary_frames(self, frames, tracks):
        if not frames:
            return frames

        fh, fw = frames[0].shape[:2]

        player_data = self._get_distances(tracks)

        team1 = sorted(
            [(pid, d['distance']) for pid, d in player_data.items()
             if d['team'] == 1 and d['distance'] > 0],
            key=lambda x: x[1], reverse=True
        )
        team2 = sorted(
            [(pid, d['distance']) for pid, d in player_data.items()
             if d['team'] == 2 and d['distance'] > 0],
            key=lambda x: x[1], reverse=True
        )

        summary = np.full((fh, fw, 3), (18, 18, 18), dtype=np.uint8)

        # ── Title ─────────────────────────────────────────────────────────────
        title = "Distance Covered"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        cv2.putText(summary, title, ((fw - tw) // 2, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

        # ── Divider ───────────────────────────────────────────────────────────
        cv2.line(summary, (fw // 2, 90), (fw // 2, fh - 30), (60, 60, 60), 2)

        # ── Column headers ────────────────────────────────────────────────────
        col1_x = fw // 4 - 160
        col2_x = fw // 4 * 3 - 160
        y_header = 130
        row_h = 48

        for label, cx in [("Team 1", col1_x), ("Team 2", col2_x)]:
            cv2.putText(summary, label, (cx, y_header),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
            cv2.putText(summary, "Player        Distance",
                        (cx, y_header + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 1)
            cv2.line(summary, (cx, y_header + 45), (cx + 380, y_header + 45),
                     (80, 80, 80), 1)

        # ── Rows ──────────────────────────────────────────────────────────────
        y_start = y_header + 80
        max_rows = (fh - y_start - 40) // row_h

        for entries, cx in [(team1, col1_x), (team2, col2_x)]:
            for i, (pid, dist) in enumerate(entries[:max_rows]):
                color = (self._RANK_COLORS[i]
                         if i < len(self._RANK_COLORS)
                         else self._DEFAULT_COLOR)
                rank_text = f"#{i + 1}"
                cv2.putText(summary, rank_text, (cx, y_start + i * row_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                cv2.putText(summary,
                            f"Player {pid:<6}  {dist:>7.1f} m",
                            (cx + 55, y_start + i * row_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        return frames + [summary] * self.SUMMARY_FRAMES
