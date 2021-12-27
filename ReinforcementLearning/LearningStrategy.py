from Models.PacManModel import PacManModel


class LearningStrategy:
    def __init__(self):
        self.model = None
        self.target_model = None

    def set_model(self, model: PacManModel):
        self.model = model
        self.target_model = model.__class__()
        self.target_model.model.set_weights(self.model.model.get_weights())

    def get_next_action(self, current_state):
        pass

    def add_record(self, old_state, action, reward, new_state, is_done):
        pass

    def end_of_episode(self):
        pass

    def beginning_of_episode(self):
        pass

    def before_action(self):
        pass

    def after_action(self, episode):
        pass

    def update_target_network(self):
        pass
