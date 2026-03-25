import torch



class Gridworld:
    def __init__(self, runs, kappa , device) -> None:
        self.device = device

        # --- ENV ---
        self.state_dim = torch.tensor((5, 8), device=self.device)
        rows = self.state_dim[0] + 1
        cols = self.state_dim[1] + 1

        self.goal_state =  torch.tensor((0, 8), device=self.device)
        self.start_state = torch.tensor(( 5, 3), device=self.device)

        self.dynamics 

        # --- end ---

        # Up | Down | Left | right |
        # 0  |  1   |  2   |   3   |
        self.actions = torch.tensor((0,1,2,3), device=self.device)
        self.no_actions = len(self.actions)
      
        self.state = torch.tile(self.start_state, (runs, 1))

        # Estimated Deterministic Model(S,A) returns S', R
        self.model = torch.full([runs, rows,cols, self.no_actions], torch.full([len(self.state_dim), 1], -1) , device=self.device) # type: ignore
        self.Q = torch.zeros((runs, rows, cols , self.no_actions), device=self.device) # type: ignore


        self.kappa = kappa  

    def train(self,episodes, planning_steps):
        for i in range(episodes):
            ...


agent = Gridworld(10, 1,"cuda")
