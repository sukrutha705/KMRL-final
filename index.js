// Modal logic
const confirmModal = document.getElementById("confirmModal");
const confirmMessage = document.getElementById("confirmMessage");
let confirmAction = null;

document.getElementById("confirmYes").addEventListener("click", async () => {
  if (confirmAction) await confirmAction();
  confirmModal.style.display = "none";
});

document.getElementById("confirmNo").addEventListener("click", () => {
  confirmModal.style.display = "none";
});

function showConfirm(message, action) {
  confirmMessage.textContent = message;
  confirmModal.style.display = "flex";
  confirmAction = action;
}
