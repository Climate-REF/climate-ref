document.addEventListener("click", function (event) {
  var button = event.target.closest("[data-flip]");
  if (button === null) {
    return;
  }
  var pair = button.parentElement.querySelector(".pair");
  if (pair !== null) {
    pair.classList.toggle("flipped");
  }
});
