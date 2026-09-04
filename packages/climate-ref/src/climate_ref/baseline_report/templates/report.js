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

document.addEventListener("click", function (event) {
  var target = event.target;
  if (!(target instanceof Element)) {
    return;
  }
  var button = target.closest("[data-header-view]");
  if (button === null) {
    return;
  }
  var wanted = button.getAttribute("data-header-view");
  var header = button.closest("[data-header]");
  if (header === null) {
    return;
  }
  header.querySelectorAll("[data-view]").forEach(function (view) {
    view.hidden = view.getAttribute("data-view") !== wanted;
  });
  header.querySelectorAll("[data-header-view]").forEach(function (other) {
    other.setAttribute("aria-pressed", String(other.getAttribute("data-header-view") === wanted));
  });
});
