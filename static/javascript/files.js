function loading() {
  $("#cover-spin").css({ display: "block" });
  var h1 = document.createElement("h1");
  h1.textContent = "Loading...";
  h1.setAttribute("class", "note");
  document.body.appendChild(h1);
}
