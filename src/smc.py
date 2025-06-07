from collections import deque
import itertools

def get_ag_knowledge(CS, ag):
	return (CS[1][ag])

class StrategyTask:
	def __init__(self, fixed_states, unchecked_states, partial_strategy):
		# F: set of states where actions are fixed
		self.F = set(fixed_states)
		# U: set of states to process
		self.U = set(unchecked_states)
		# SC: dict mapping state -> action-tuple for coalition C
		self.SC = dict(partial_strategy)

	def copy(self):
		return StrategyTask(self.F, self.U, self.SC)

	@staticmethod
	def children_strategies(G, SC, CS, coalition, game):
		"""
		Implements the pseudocode:

		for each agent a in coalition:
		if SC is already defined on [CS]~a then
			possibleActions[a] = {that fixed action}
		else
			possibleActions[a] = all actions a can take in any state
								indistinguishable from CS

		Then builds one new SC per joint‐action combination,
		assigning the chosen action to *every* state in the info‐set.
		Returns a list of new partial‐strategy dicts (excluding the old SC).
		"""

		per_agent = []
		# 1) For each agent determine their local action‐choices
		for a in coalition:
			# compute the information‐set [CS]~a
			ag_know = get_ag_knowledge(CS, a)
			# get states where agent has same knowledge state
			# i.e if joint knowledge is (AG0, AG1) only the corresponding knowledge must match
			# so indistinguishable states with respect to agents knowledge
			indist_states = [
				q for q in G.nodes()
				if q[1][a] == ag_know
			]

			# see if SC already fixes an action for any state in that info‐set
			fixed = set()
			for q in indist_states: 
				if q in SC:
					fixed = SC[q][a]
			if fixed:
				# uniformity: only that one choice
				per_agent.append([fixed])
			else:
				# collect *all* actions that a can perform in any q in indist_states
				local = {
					data['joint_action'][a]
					for _,_,data in G.out_edges(CS, data=True)
				}
				per_agent.append(list(local))

		# 2) Cartesian‐product over agents to get all joint‐actions
		all_joints = itertools.product(*per_agent)

		children = []
		for joint in all_joints:
			# build a new SC by copying the old one
			newSC = dict(SC)

			# assign `joint` to *every* state in each agent’s info‐set
			# try doing this implicitly instead
			newSC[CS] = joint
			# only keep truly new strategies
			if newSC != SC:
				children.append(newSC)
		return children

	@staticmethod
	def next_states(G, SC, CS, coalition):
		"""
		Implements the pseudocode:

		Let next = ∅;
		Let allActions = d1([CS]~1)×…×dk([CS]~k);
		for each joint ∈ allActions:
			if joint not conflicting with SC(CS):
			next := next ∪ { δ(CS, joint) }
		return next

		Here we assume G is a MultiDiGraph whose edges from CS
		carry data['joint_action'] = full joint-action tuple.
		SC[CS] (if defined) is the coalition’s projection of that tuple.
		"""
		succs = set()
		# See if we already fixed a coalition action at CS
		fixed = SC.get(CS, None)
	
		for _, v, data in G.out_edges(CS, data=True):
			joint = data['joint_action']
			# If we’ve fixed coalition’s move at CS, skip conflicting edges
			if fixed is not None:
				if tuple(joint[i] for i in coalition) != fixed:
					continue
			# Otherwise (or if no fixed action), this edge is allowed
			succs.add(v)
		return succs

	@staticmethod
	def synthesize_strategy(
		G,
		initial_state,
		coalition,
		check_strategy,
		game,
		only_path_based=False,
	):
		"""
		G: epistemic MultiDiGraph
		initial_state: (location, knowledge_tuple)
		coalition: list of agent indices
		check_strategy: fn(G, SC, initial_state, coalition) -> bool
		game: Game object for observation partitions
		only_path_based: if True, only follow the first child at each branch
		"""
		
		# 0)
		STL = deque([ StrategyTask(set(), {initial_state}, {}) ])

		# 1.
		while STL:
			task = STL.popleft() #fifo
			F, U, SC = task.F, task.U, task.SC
			# 1) If no unchecked, verify and return if good
			if not U:
				if check_strategy(G, SC, initial_state):
					return SC
				continue
			# 2) Ta första 
			CS = U.pop()      # remove it from U
			F.add(CS)
			newStrategies = StrategyTask.children_strategies(G, SC, CS, coalition, game)
			# 3) Build all possible child‐SC by fixing CS
			if not newStrategies:
				# no new strategies → requeue with the original SC
				if U:
					newU = U.union(StrategyTask.next_states(G, SC, CS, coalition) - F)
					STL.append( StrategyTask(F.copy(), newU, SC.copy()) )
			else:
				for newS in newStrategies:
					STL.append( StrategyTask(F.copy(), (U | (StrategyTask.next_states(G, SC, CS, coalition)) - F), newS))

			# 6) Conditional verification (we skip unbounded/complete here)
			checkCurrent = True
			if not newStrategies:
				checkCurrent = False
			if checkCurrent:
				# We verify *after* branching on the “current” strategy
				# i.e. the one we just enqueued as first
				# Its partial‐SC is always the *last* one we inserted:
				#   if newStrategies: it's `first`, else it's the same SC
				cand = SC
				if check_strategy(G, cand, initial_state):
					return cand
		# 7) Exhausted all tasks
		return None






