from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.left_team_id = 1
        self.right_team_id = 2

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0] / 2), :]
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        player_x_positions = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_colors.append(self.get_player_color(frame, bbox))
            player_x_positions.append((bbox[0] + bbox[2]) / 2)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Determine which cluster/team defends the left vs right side of the field
        labels = kmeans.labels_
        xs_0 = [player_x_positions[i] for i, l in enumerate(labels) if l == 0]
        xs_1 = [player_x_positions[i] for i, l in enumerate(labels) if l == 1]
        mean_x_0 = sum(xs_0) / len(xs_0) if xs_0 else float('inf')
        mean_x_1 = sum(xs_1) / len(xs_1) if xs_1 else float('inf')

        if mean_x_0 < mean_x_1:
            self.left_team_id, self.right_team_id = 1, 2
        else:
            self.left_team_id, self.right_team_id = 2, 1

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        # Goalkeepers wear different jersey colors so KMeans misclassifies them.
        # Players near the goal edges of the frame are almost always goalkeepers —
        # use their x-position to assign the correct team instead.
        frame_width = frame.shape[1]
        player_center_x = (player_bbox[0] + player_bbox[2]) / 2
        if player_center_x < frame_width * 0.15:
            team_id = self.left_team_id
        elif player_center_x > frame_width * 0.85:
            team_id = self.right_team_id

        self.player_team_dict[player_id] = team_id
        return team_id
