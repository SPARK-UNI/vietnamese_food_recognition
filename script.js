const predictBtn = document.getElementById("predictBtn");
const imageInput = document.getElementById("imageInput");
const startCamBtn = document.getElementById("startCamBtn");
const captureBtn = document.getElementById("captureBtn");
const webcam = document.getElementById("webcam");
const previewImg = document.getElementById("previewImg");
const predictionText = document.getElementById("prediction");
const dropZone = document.getElementById("dropZone");

let currentBase64Image = null;

// Hàm gửi ảnh base64 tới API /predict
async function sendToServer(base64Image) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64Image })
  });
  const data = await response.json();
  predictionText.textContent = "Kết quả: " + data.prediction;
}

// Xử lý hiển thị ảnh
function showPreview(file) {
  const reader = new FileReader();
  reader.onload = () => {
    currentBase64Image = reader.result;
    previewImg.src = currentBase64Image;
    predictionText.textContent = "Chưa có kết quả";
  };
  reader.readAsDataURL(file);
}

// Khi bấm dropZone thì mở file dialog
dropZone.addEventListener("click", () => imageInput.click());

// Khi chọn file bằng input
imageInput.addEventListener("change", () => {
  if (imageInput.files.length > 0) {
    showPreview(imageInput.files[0]);
  }
});

// Drag & Drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.background = "#d0e6f7";
});
dropZone.addEventListener("dragleave", () => {
  dropZone.style.background = "#ecf0f1";
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.background = "#ecf0f1";
  if (e.dataTransfer.files.length > 0) {
    showPreview(e.dataTransfer.files[0]);
  }
});

// Predict khi bấm nút
predictBtn.addEventListener("click", () => {
  if (!currentBase64Image) {
    alert("Chưa có ảnh để predict!");
    return;
  }
  sendToServer(currentBase64Image);
});

// Bật webcam
startCamBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
    webcam.hidden = false;
    captureBtn.hidden = false;
    startCamBtn.hidden = true;
  } catch (err) {
    console.error("Không mở được webcam:", err);
  }
});

// Capture từ webcam
captureBtn.addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  canvas.width = webcam.videoWidth;
  canvas.height = webcam.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

  currentBase64Image = canvas.toDataURL("image/jpeg");
  previewImg.src = currentBase64Image;
  predictionText.textContent = "Chưa có kết quả";
});
