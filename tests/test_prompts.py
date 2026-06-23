from agentix import PromptRegistry


def test_register_get_and_versioning() -> None:
    r = PromptRegistry()
    assert r.register("sys", "v one text") == 1
    assert r.register("sys", "v two text") == 2
    assert r.get("sys") == "v two text"          # latest is active
    assert r.get("sys", version=1) == "v one text"
    assert r.versions("sys") == [1, 2]
    assert r.active("sys") == 2
    assert "sys" in r and r.names() == ["sys"]


def test_register_identical_is_a_noop() -> None:
    r = PromptRegistry()
    r.register("p", "same")
    assert r.register("p", "same") == 1  # no new version created
    assert r.versions("p") == [1]


def test_rollback() -> None:
    r = PromptRegistry()
    r.register("p", "good")     # v1
    r.register("p", "bad")      # v2 (regressed)
    assert r.get("p") == "bad"
    r.rollback("p", 1)
    assert r.get("p") == "good"
    assert r.active("p") == 1
    # registering again continues numbering from the latest, not the active one
    assert r.register("p", "fixed") == 3
    assert r.get("p") == "fixed"


def test_render_formats() -> None:
    r = PromptRegistry()
    r.register("greet", "Hello, {name}.")
    assert r.render("greet", name="Sam") == "Hello, Sam."


def test_missing_prompt_and_version_raise() -> None:
    r = PromptRegistry()
    for call in (lambda: r.get("nope"), lambda: r.versions("nope")):
        try:
            call()
            raise AssertionError("expected KeyError")
        except KeyError:
            pass
    r.register("p", "x")
    try:
        r.get("p", version=99)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    try:
        r.rollback("p", 99)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_persistence_round_trip() -> None:
    r = PromptRegistry()
    r.register("a", "a1")
    r.register("a", "a2")
    r.rollback("a", 1)
    r.register("b", "b1")

    restored = PromptRegistry.from_dict(r.to_dict())
    assert restored.get("a") == "a1"        # active rolled-back version preserved
    assert restored.get("a", version=2) == "a2"
    assert restored.get("b") == "b1"
    assert restored.versions("a") == [1, 2]
