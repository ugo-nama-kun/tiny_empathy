import pytest
import numpy as np
from tiny_empathy.envs.double_foodshare_pz import DoubleFoodShareEnvPZ, AgentActions


@pytest.fixture
def env():
    return DoubleFoodShareEnvPZ(render_mode=None)


def test_environment_initialization(env):
    assert env.possible_agents == ["agent0", "agent1"]
    assert isinstance(env.observation_space("agent0").shape, tuple)


def test_reset_returns_correct_structure(env):
    observations, infos = env.reset()
    assert isinstance(observations, dict)
    assert isinstance(infos, dict)
    assert all(agent in observations for agent in env.possible_agents)
    assert all(agent in infos for agent in env.possible_agents)
    for obs in observations.values():
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2,)  # cognitive_empathy is False by default


def test_step_function_changes_state(env):
    env.reset()
    actions = {agent: AgentActions.eat for agent in env.possible_agents}
    obs, rewards, dones, truncateds, infos = env.step(actions)

    assert isinstance(obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(dones, dict)
    assert isinstance(truncateds, dict)
    assert isinstance(infos, dict)

    for agent in env.possible_agents:
        assert agent in obs
        assert isinstance(rewards[agent], float)
        assert isinstance(dones[agent], bool)
        assert isinstance(truncateds[agent], bool)


def test_empathy_observation_shape():
    env = DoubleFoodShareEnvPZ(cognitive_empathy=True)
    observations, _ = env.reset()
    for obs in observations.values():
        assert obs.shape == (3,)  # 3-dim with cognitive empathy


def test_energy_updates_on_eat_action(env):
    env.reset()
    food_owner = env.food_owner
    owner_agent = env.possible_agents[food_owner]
    non_owner = env.possible_agents[1 - food_owner]

    env.set_agent_info(owner_agent, energy=0.0, have_food=True)
    env.set_agent_info(non_owner, energy=0.0, have_food=False)

    actions = {
        owner_agent: AgentActions.eat,
        non_owner: AgentActions.protect
    }

    _, _, _, _, _ = env.step(actions)
    assert env.agent_info[owner_agent]["energy"] > 0
    assert env.agent_info[non_owner]["energy"] <= 0


def test_energy_updates_on_protect_action(env):
    env.reset()
    food_owner = env.food_owner
    owner_agent = env.possible_agents[food_owner]
    non_owner = env.possible_agents[1 - food_owner]

    env.set_agent_info(owner_agent, energy=0.0, have_food=True)
    env.set_agent_info(non_owner, energy=0.0, have_food=False)

    actions = {
        owner_agent: AgentActions.protect,
        non_owner: AgentActions.eat
    }

    _, _, _, _, _ = env.step(actions)
    assert env.agent_info[owner_agent]["energy"] <= 0
    assert env.agent_info[non_owner]["energy"] <= 0


def test_energy_updates_on_share_eat(env):
    env.reset()
    food_owner = env.food_owner
    owner_agent = env.possible_agents[food_owner]
    non_owner = env.possible_agents[1 - food_owner]

    env.set_agent_info(owner_agent, energy=0.0, have_food=True)
    env.set_agent_info(non_owner, energy=0.0, have_food=False)

    actions = {
        owner_agent: AgentActions.share,
        non_owner: AgentActions.eat
    }

    _, _, _, _, _ = env.step(actions)
    assert env.agent_info[owner_agent]["energy"] <= 0
    assert env.agent_info[non_owner]["energy"] > 0


def test_energy_updates_on_share_share(env):
    env.reset()
    food_owner = env.food_owner
    owner_agent = env.possible_agents[food_owner]
    non_owner = env.possible_agents[1 - food_owner]

    env.set_agent_info(owner_agent, energy=0.0, have_food=True)
    env.set_agent_info(non_owner, energy=0.0, have_food=False)

    actions = {
        owner_agent: AgentActions.share,
        non_owner: AgentActions.share
    }

    _, _, _, _, _ = env.step(actions)
    # Food transfer
    assert env.agent_info[owner_agent]["energy"] <= 0
    assert env.agent_info[owner_agent]["have_food"] is False
    assert env.agent_info[non_owner]["energy"] <= 0
    assert env.agent_info[non_owner]["have_food"] is True


def test_episode_termination_on_energy_extreme():
    env = DoubleFoodShareEnvPZ()
    env.reset()
    agent = env.possible_agents[0]
    env.agent_info[agent]["energy"] = 1.1  # Set out-of-bound energy

    actions = {a: AgentActions.protect for a in env.possible_agents}
    _, _, dones, _, _ = env.step(actions)

    assert all(dones.values())


def test_max_episode_length_termination():
    env = DoubleFoodShareEnvPZ()
    env.reset()
    env._step = env.max_episode_length - 1

    actions = {a: AgentActions.protect for a in env.possible_agents}
    _, _, dones, _, _ = env.step(actions)

    assert all(dones.values())
