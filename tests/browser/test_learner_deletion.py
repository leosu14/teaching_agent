"""MVP item 9 in a real browser: the privacy notice, and a learner deleting their data from the student app.

The learner is the learning-cycle fixture's, mid-cycle, so there is everything to delete (a goal and its plan, a
lesson with its files, a teaching session, mastery, a cycle). The deletion takes two steps on the Progress page and
then the tab forgets the learner; coming back starts from the welcome screen. While a lesson is being prepared the
page says so and nothing is deleted."""

from __future__ import annotations

from tests.browser.test_student_app import page_for, site  # noqa: F401 - the student app's fixtures
from tests.learning_cycle_fixtures import LEARNER


def test_the_privacy_notice_is_linked_and_needs_no_script(container, site, page_for) -> None:  # noqa: F811
    url, token = site(container, "web-privacy")
    page = page_for(url, token)
    page.get_by_test_id("onboarding-privacy").wait_for()  # linked before the learner gives any detail
    page.get_by_test_id("privacy-link").click()  # the footer link
    page.locator("h1", has_text="Privacy notice").wait_for()
    text = page.locator("main").inner_text()
    for heading in ("Who runs this service", "What is kept about you", "Who can see it", "AI and media services",
                    "How long it is kept", "Deleting your data"):
        assert heading in text, heading
    assert page.locator("script").count() == 0
    page.get_by_role("link", name="Back to the app").click()
    page.get_by_test_id("display-name").wait_for()
    assert page.problems == []


def test_a_learner_deletes_their_data_from_the_progress_page(cycle_env_at, site, page_for) -> None:  # noqa: F811
    env = cycle_env_at("session")
    repo = env.container.learner_deletion.repository
    assert repo.holds_data(LEARNER)
    url, token = site(env.container, LEARNER)
    page = page_for(url, token)
    page.locator("nav a[data-nav=progress]").click()
    card = page.get_by_test_id("your-data")
    card.wait_for()
    assert card.get_by_test_id("privacy-notice").get_attribute("href") == "privacy.html"
    assert not page.get_by_test_id("delete-data").is_visible()  # behind a first step

    page.get_by_test_id("open-delete").click()
    page.get_by_test_id("confirm-delete").fill("delete me")
    page.get_by_test_id("delete-data").click()
    assert "Type DELETE to confirm." in card.get_by_test_id("error").inner_text()
    assert repo.holds_data(LEARNER)  # nothing was sent

    page.get_by_test_id("confirm-delete").fill("DELETE")
    page.get_by_test_id("delete-data").click()
    page.get_by_test_id("deleted").wait_for()
    assert page.locator("h1").inner_text() == "Your data has been deleted"
    assert page.locator("#nav").is_hidden()
    assert not repo.holds_data(LEARNER)

    page.get_by_role("button", name="Start again").click()
    page.get_by_test_id("display-name").wait_for()  # a new start: the welcome screen
    assert page.problems == []


def test_a_lesson_in_preparation_blocks_the_deletion(cycle_env_at, site, page_for) -> None:  # noqa: F811
    env = cycle_env_at("session")
    c = env.container
    c.task_service.enqueue_task(request="Teach me Spanish", learner_id=LEARNER, user_id="web")  # queued, not run
    url, token = site(c, LEARNER)
    page = page_for(url, token)
    page.goto(f"{url}/app/#/progress")
    page.get_by_test_id("open-delete").click()
    page.get_by_test_id("confirm-delete").fill("DELETE")
    page.get_by_test_id("delete-data").click()
    error = page.get_by_test_id("your-data").get_by_test_id("error")
    error.wait_for()
    assert "A lesson is being prepared right now" in error.inner_text()
    assert page.get_by_test_id("delete-data").is_enabled()  # the learner can try again later
    assert c.learner_deletion.repository.holds_data(LEARNER)
    assert page.problems == []
