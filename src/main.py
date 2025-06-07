from cuplifting import build_game
from smc import StrategyTask
from collections import deque
import networkx as nx

def existential_reachability_check(G, SC, initial_state, is_goal):
	return rec_search(G, SC, initial_state, is_goal)


def rec_search(G, SC, initial_state, is_goal):
	visited = set()
	queue = deque([initial_state])
	while queue:
		cur = queue.popleft()
		if cur in visited:
			return False
		visited.add(cur)

		if is_goal(cur) and not queue:
			return True
		if cur in SC:
			counter = 0
			for (_, v, data) in G.out_edges(cur, data=True):
				if data["joint_action"] == SC[cur]:
					if counter == 0:
						queue.append(v)
					else:
						# If branch is not winning, return false
						if not rec_search(G, SC, v, is_goal):
							return False
					counter += 1
		else:
			return False
	return False
		
def is_goal(state):
	loc, _ = state
	return loc == 4  # or your actual winning location


def check_strategy(G, SC, initial_state, game=None):
	# game isn’t needed here; we only need G, SC, initial_state, coalition
	return existential_reachability_check(
		G, SC, initial_state, is_goal
	)

def main():
	# 1) Build the Game scenario
	game = build_game()

	# 2) Construct the knowledge (epistemic) graph and initial state
	G, initial_state = game.graph_search()

	# 3) Define the coalition of all agents (or subset as needed)
	coalition = game.agents

	# # 4) Invoke the SMC algorithm
	winner = StrategyTask.synthesize_strategy(
		G               = G,
		initial_state   = initial_state,
		coalition       = coalition,
		check_strategy  = check_strategy,
		game            = game,
		only_path_based = False
	)

	# # 5) Report the result
	if winner:
		print("Found winning strategy:")
		for state, action in winner.items():
			print(f"  At {state[1]} -> {action}")
	else:
		print("No winning strategy exists.")

if __name__ == "__main__":
	main()
