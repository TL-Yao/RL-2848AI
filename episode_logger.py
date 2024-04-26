class EpisodeLog:
    def __init__(self, total_episode):
        self.total_episode = total_episode
        self.loss = []
        self.gradients = []
        self.experience = []
