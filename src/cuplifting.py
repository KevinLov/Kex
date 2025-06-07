from knowledge import Game

def build_game():
    locations = [0, 1, 2, 3, 4]
    agent_obs = ([[0], [1], [2], [3], [4]], [[0], [1, 2], [3], [4]])
    agent_actions = ({"g", "s", "l"}, {"g", "s", "l"})
    start_location = 0
    agents = [0, 1]
    transitions = [
        # from start (0)
        (0, ("g","g"), 1),  # start g,g → bad
        (0, ("g","g"), 2),  # start g,g → good

        # from bad (1)
        (1, ("l","l"), 3),  # bad l,l → lose
        (1, ("s","l"), 3),  # bad s,l → lose
        (1, ("l","s"), 3),  # bad l,s → lose
        (1, ("s","s"), 2),  # bad s,s → good

        # from good (2)
        (2, ("s","l"), 3),  # good s,l → lose
        (2, ("l","s"), 3),  # good l,s → lose
        (2, ("s","s"), 2),  # good s,s → good
        (2, ("l","l"), 4),  # good l,l → win

        # absorbing
        (3, ("s","s"), 3),  # lose loops
        (4, ("l","l"), 4),  # win loops
    ]
    knowledge_levels = {
        0: 2,
        1: 1
    }

    return Game(
        locations=locations,
        start_location=start_location,
        agents=agents,
        transistions=transitions,
        agent_actions=agent_actions,
        agent_obs=agent_obs,
        knowledge_levels=knowledge_levels
    )
