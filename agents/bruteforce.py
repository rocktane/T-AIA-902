from agents import BaseAgent

class Bruteforce(BaseAgent):
    def choose_action(self, state):
        return self.env.action_space.sample()
