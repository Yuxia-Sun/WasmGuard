(function () {
  document.addEventListener("click", (e) => {
    const target = e.target;
    switch (target.id) {
      case "sendData":
        const input1 = document.getElementById("input1");
        chrome.tabs.query(
          { active: true, currentWindow: true },
          function (tab) {
            chrome.tabs.sendMessage(
              tab[0].id,
              {
                action: "hello",
                data: input1.value,
              },
              function (response) {
                alert("收到回复：" + response.state);
              }
            );
          }
        );
        break;
    }
  });
})();
