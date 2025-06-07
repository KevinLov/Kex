import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class Game:
	# locations is a set of locations
	# start_location is a single location
	# agents is a list of numbers representing the agents
	# agent_actions is a tuple with each member being a set of actions available to a specific agent
	# joint_actions is the cartesian product of all agent_actions
	# transistions is a set of	 possible transistions as (location, joint_action, location) tuples
	# agent_obs is a tuple with size #agent, containing partitions of locations 
	# giving the possible observations for agent i
	# joint_obs is the cartesian product of all agent_obs
	# knowledge_levels is a dictionary from agents to knowledge levels

	def __init__(self, locations, start_location, agents, transistions, agent_actions, agent_obs, knowledge_levels):
		self.locations = locations
		self.start_location = start_location
		self.agents = agents
		self.agent_actions = agent_actions
		self.transistions = transistions 
		self.joint_actions = set()
		for (_, joint_act, _) in transistions:
			self.joint_actions.add(joint_act)
		self.joint_actions = tuple(self.joint_actions)
		self.agent_obs = agent_obs
		temp = itertools.product(*agent_obs)
		self.joint_obss = []
		# Remove impossible observations
		for obs in temp:
			if list(set(obs[0]) & set(obs[1])):
				self.joint_obss.append(obs)
		self.joint_obss = tuple(self.joint_obss)
		self.graph = nx.MultiDiGraph()
		self.knowledge_levels = knowledge_levels
		for transistion in self.transistions:
			self.graph.add_edge(transistion[0], transistion[2], joint_action=transistion[1])

	def graph_search(self):
		knowledge_graph = nx.MultiDiGraph()

		# Vi sparar kunskapen som en tuple av tupler där varje inre tubel visar kunskapen för en specifik agent.
		initial_knowledge = []
		for a in self.agents:
			start_know = tuple(self.get_obs_block(a, self.start_location))
			if self.knowledge_levels[a] == 2:
				initial_knowledge.append(tuple([tuple([start_know, start_know])]))
			else:
				initial_knowledge.append(start_know)
		# Konvertera till tuple
		initial_knowledge = tuple(initial_knowledge)

		start_state = (self.start_location, initial_knowledge)
		knowledge_graph.add_node(start_state)

		queue = deque([start_state])
		visited =     { start_state }

		while queue:
			loc, know = queue.popleft()
			for nbr in self.graph.neighbors(loc):
				for _, data in self.graph[loc][nbr].items():
					
					joint = data["joint_action"]
					
					# Uppdatera kunskap per agent
					new_know = tuple(
						self.nknowledge_update(
							agent=i,
							knowledge=know[i],
							action=joint[i],
							observation=self.get_obs_block(i, nbr),
							level=self.knowledge_levels[i]
						)
						for i in self.agents
					)

					new_state = (nbr, new_know)
					if new_state not in visited:
						visited.add(new_state)
						queue.append(new_state)
					# Lägg alltid till kanten
					knowledge_graph.add_edge(
						(loc, know),
						new_state,
						joint_action=joint
					)
		pos = nx.spring_layout(knowledge_graph, k=1.0, iterations=40, seed=42)

		nx.draw(
            knowledge_graph, pos,
            with_labels=True,
            node_size=1200,
            node_color="lightblue",
            font_size=8,
            arrowsize=10
        )

		edge_labels = {
            (u, v): f"({data['joint_action'][0]}, {data['joint_action'][1]})"
            for u, v, data in knowledge_graph.edges(data=True)
        }

		nx.draw_networkx_edge_labels(
            knowledge_graph, pos,
            edge_labels=edge_labels,
            font_size=6,
            label_pos=0.5
        )
		plt.axis("off")
		plt.tight_layout()
		plt.show()

		return knowledge_graph, start_state
		
	def knowledge_update(self, agent, knowledge, action, observation):
		new_knowledge = []
		for (location, joint_action, new_location) in self.transistions:
			if location in knowledge and joint_action[agent] == action and new_location in observation:
				new_knowledge.append(new_location)
		return tuple(sorted(set(new_knowledge)))

	def nknowledge_update(self, agent, knowledge, action, observation, level): 
		# Note: Knowledge needs to be of structure ((a, b)) or ((a, b), (a, b)) for this
		# ag_knowledge should be a "k-1" knowledge structure
		if level == 1:
			return self.knowledge_update(agent, knowledge, action, observation)
		def inner_loop(ag):
			new_knowledge_ag = []
			# Usually len(poss_knowledge) is 1 but could be more
			for ag_knowledge in knowledge:
				for joint_obs in self.joint_obss:
					if joint_obs[agent] == observation:
						for joint_action in self.joint_actions:
							if joint_action[agent] == action:
								res = self.nknowledge_update(ag, ag_knowledge[ag], joint_action[ag], joint_obs[ag], level-1)
								if res != ():
									new_knowledge_ag.append(res)
			return tuple(set(new_knowledge_ag))
		agent_knowledge = []	
		for ag in self.agents: 
			agent_knowledge.append(inner_loop(ag))

		new_knowledge = itertools.product(*agent_knowledge)
		new_knowledge = list(new_knowledge)

		# Prune inconsistent tuples; any states where the intersection of knowledge between agents is empty
		pruned_knowledge = new_knowledge.copy()
		for state in new_knowledge:
			for ag1, ag2 in itertools.combinations(self.agents, 2):
				if not set(state[ag1]) & set(state[ag2]):
					pruned_knowledge.remove(state)
		
		new_knowledge = pruned_knowledge

		
		return tuple(new_knowledge)
	
	def get_obs_block(self, agent: int, location: int):
		for block in self.agent_obs[agent]:
			if location in block:
				return block
		raise ValueError("No obs‐block!")
