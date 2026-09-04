document.addEventListener("click", function (event) {
  var target = event.target;
  if (!(target instanceof Element)) {
    return;
  }
  var button = target.closest("[data-flip]");
  if (button === null) {
    return;
  }
  var figure = button.closest("figure");
  var pair = figure === null ? null : figure.querySelector(".pair");
  if (pair !== null) {
    pair.classList.toggle("flipped");
  }
});
