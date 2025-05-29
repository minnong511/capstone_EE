document.addEventListener("DOMContentLoaded", () => {
    function showNotification(text) {
      const noti = document.createElement('div');
      noti.className = "px-4 py-2 rounded bg-red-500 text-white shadow";
      noti.innerText = text;
  
      const container = document.getElementById("notification-area");
      if (container) {
        container.appendChild(noti);
  
        setTimeout(() => {
          noti.remove();
        }, 5000); // 5초 후 자동 제거
      }
    }
  
    fetch('/alerts')
      .then(res => res.json())
      .then(data => {
        data.slice(0, 5).forEach(alert => {
          const message = `[${alert.room_id}] ${alert.category}`;
          showNotification(message);
        });
      });
  });