from schoolsched_demo.mock_data import build_demo_dataset, generate_plan_set, preset_to_config


def test_generate_mock_plan_set():
    dataset = build_demo_dataset()
    config = preset_to_config(dataset.scenario_presets["春季标准周排课"])
    plans = generate_plan_set(dataset, config)

    assert len(plans) == 4
    assert {plan.plan_id for plan in plans} == {"balance", "income", "open", "resource"}
    assert all(plan.metrics.total_requests == dataset.total_requests for plan in plans)
    assert all(plan.metrics.revenue > 0 for plan in plans)
