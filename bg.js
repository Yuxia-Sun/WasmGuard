// 对当前页面插入一些页面

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if(message.type === "vue_page_mounted") {
    console.log("vue页面已建立")
  }
});

function appendDiv() {
  const chromePlugPanelApp = document.createElement("div");
  chromePlugPanelApp.id = "chromePlugPanelApp";
  document.body.appendChild(chromePlugPanelApp);
  console.log("插件面板已插入");
  fastKeyListen();
}
function fastKeyListen() {
  const keydownFn = (event) => {
    if (event.altKey && event.key === "v") {
      const chromePlugPanel = document.getElementById("chromePlugPanelApp");
      console.log("chromePlugPanel: ", chromePlugPanel)
      chromePlugPanel.style.display =
        chromePlugPanel.style.display === "none" ? "block" : "none";
    }
  };
  const chromePlugPanel = document.getElementById("chromePlugPanelApp");
  document.addEventListener("keydown", keydownFn);
  chromePlugPanel.addEventListener("keydown", keydownFn);
}
appendDiv();

// 监听来自background的消息
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  // 将消息发送到Vue页面
  console.log("bg.js收到信息", message);
  if(message.type === "data_to_show") {
    console.log("收到来自background的消息，转发给vue页面")
    window.postMessage(message, "*");
  }
});


// 中转代码，主要用于转发
window.addEventListener("message", (event) => {
  // 验证消息来源等安全考虑省略
  if (event.source == window && event.data.type && event.data.type == "FROM_INJECTED_SCRIPT") {
    // 将消息转发到background script
    chrome.runtime.sendMessage(event.data);
    return
  }

  if (event.source === window && event.data.type === "WASM_FOUND") {
      chrome.runtime.sendMessage(event.data);
      return 
  }

  if (event.source === window && event.data.type === "WASM_COMPLETE") {
      chrome.runtime.sendMessage(event.data);
      return
  }
  //console.log(event.source)
  // vue页面创建完毕
  if(event.data.type === "vue_page_mounted") {
    console.log(event.source)
    chrome.runtime.sendMessage({
      type:"vue_page_mounted"
    })
  }

  if (event.data.action === 'closeTab') {  
    // 执行关闭标签页或其他相关操作  
    console.log("快关闭网页!")
    chrome.tabs?.query({ active: true, currentWindow: true }, function(tabs) {  
        chrome.tabs?.remove(tabs[0].id);  
    });  
  }   

});