import numpy as np

from .utils.core import Agent, Landmark, Wall, World
from .utils.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        num_row_walls = 2
        num_col_walls = 2
        num_row_p_walls = 12
        num_col_p_walls = 12
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.02
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02
        # add walls
        r_walls = [Wall() for _ in range(num_row_walls)]
        c_walls = [Wall() for _ in range(num_col_walls)]
        r_p_walls = [Wall() for _ in range(num_row_p_walls)]
        c_p_walls = [Wall() for _ in range(num_col_p_walls)]
        
        for i, wall in enumerate(r_walls):
            wall.name = 'row wall %d' % i
            wall.collide = True
            wall.movable = False
            wall.pos = [[-3, 0], [3, 0]] 
        for i, wall in enumerate(c_walls):
            wall.name = 'col wall %d' % i
            wall.collide = True
            wall.movable = False
            wall.pos = [[0, -3], [0, 3]]
        for i, wall in enumerate(r_p_walls):
            wall.name = 'row p wall %d' % i
            wall.collide = True
            wall.movable = False
            wall.pos = [[-0.3, 0], [0.3, 0]]
        for i, wall in enumerate(c_p_walls):
            wall.name = 'col p wall %d' % i
            wall.collide = True
            wall.movable = False
            wall.pos = [[0, -0.3], [0, 0.3]]

        world.walls = r_walls + c_walls + r_p_walls + c_p_walls
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.45])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-2, +2, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-2, +2, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # set wall positions
        WallPos = [[0,-3], [0, 3], [-3, 0], [3, 0],
                   [-2.7, 1], [-1.3, 1], [-0.7, 1], [0.7, 1], [1.3, 1], [2.7, 1],
                   [-2.7, -1], [-1.3, -1], [-0.7, -1], [0.7, -1], [1.3, -1], [2.7, -1],
                   [-1, -2.7], [-1, -1.3], [-1, -0.7], [-1, 0.7], [-1, 1.3], [-1, 2.7],
                   [1, -2.7], [1, -1.3], [1, -0.7], [1, 0.7], [1, 1.3], [1, 2.7]]
        for i, wall in enumerate(world.walls):
            wall.state.p_pos = np.array(WallPos[i])
            wall.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
                if self.is_boundary(a, world):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def is_boundary(self, agent, world):
        for wall in world.walls:
            agent_pos = agent.state.p_pos
            wall_pos = wall.state.p_pos
            wall_att = wall.state.p_pos
            row, _ = wall_att[0][1] > 0, wall_att[0][0] > 0
            dist = agent_pos[0] - wall_pos[0] if row else agent_pos[1] - wall_pos[1]
            dist_min = agent.size + 2
            
            if dist < dist_min:
                return True
            else:
                continue
        return False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def global_reward(self, world):
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
