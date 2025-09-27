window.onload = function() {
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const cameraBtn = document.getElementById("cameraBtn");
  const captureBtn = document.getElementById("captureBtn");
  const camera = document.getElementById("camera");
  const canvas = document.getElementById("canvas");
  const preview = document.getElementById("preview");
  const result = document.getElementById("result");
  let stream;

  // Hiển thị ảnh preview khi chọn file
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = e => {
          preview.innerHTML = `<img src="${e.target.result}">`;
        };
        reader.readAsDataURL(file);
      }
    });
  }

  // Upload ảnh để predict
  if (uploadBtn) {
    uploadBtn.addEventListener("click", () => {
      const file = fileInput.files[0];
      if (!file) {
        alert("Hãy chọn ảnh!");
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => {
        fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: reader.result })
        })
          .then(res => res.json())
          .then(data => {
            result.innerText = `🍽️ Món ăn: ${data.prediction}`;
          });
      };
      reader.readAsDataURL(file);
    });
  }

  // Mở camera
  if (cameraBtn) {
    cameraBtn.addEventListener("click", async () => {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      camera.srcObject = stream;
      captureBtn.style.display = "inline-block";
    });
  }

  // Chụp ảnh từ camera
  if (captureBtn) {
    captureBtn.addEventListener("click", () => {
      const ctx = canvas.getContext("2d");
      canvas.width = camera.videoWidth;
      canvas.height = camera.videoHeight;
      ctx.drawImage(camera, 0, 0);

      const dataURL = canvas.toDataURL("image/jpeg");
      preview.innerHTML = `<img src="${dataURL}">`;

      fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL })
      })
        .then(res => res.json())
        .then(data => {
          result.innerText = `🍽️ Món ăn: ${data.prediction}`;
        });
    });
  }
};
