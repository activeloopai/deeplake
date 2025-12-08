document.addEventListener("DOMContentLoaded", function () {
  let slackButton = document.createElement("nav");
  slackButton.classList.add("md-header-nav__button", "md-button");

  // Create the <a> element
  let slackLink = document.createElement("a");
  slackLink.href =
    "https://slack.activeloop.ai";
  slackLink.target = "_blank"; // Open in a new tab
  slackLink.title = "Join us on Slack";

  let slackIcon = document.createElement("img");
  slackIcon.src = "/images/slack.svg";
  slackIcon.alt = "Slack";
  slackButton.setAttribute("target", "_blank");
  slackIcon.style.height = "20px";
  slackIcon.style.height = "20px";
  slackIcon.style.width = "20px";
  slackIcon.style.marginRight = "8px";

  slackLink.appendChild(slackIcon);

  slackButton.appendChild(slackLink);

  let navBar = document.querySelector(".md-header__source");
  if (navBar) {
    navBar.parentNode.insertBefore(slackButton, navBar.nextSibling);
  }

  let nextSteps = document.getElementById("next-steps");
  if (nextSteps) {
    let nextStepsIcon = document.createElement("img");
    nextStepsIcon.src = "/images/forward.svg";
    nextStepsIcon.alt = "Next Steps Icon";
    nextStepsIcon.style.height = "20px";
    nextStepsIcon.style.width = "20px";
    nextStepsIcon.style.marginLeft = "8px";

    // Append the icon to the #next-steps div
    nextSteps.appendChild(nextStepsIcon);
  }
});
