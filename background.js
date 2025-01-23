const detect_server = "http://8.138.28.248:8201/upload"
const result_map = new Map(); // 存储后端返回结果
const wasm_map = new Map(); // 存储注入脚本发送的wasm
const host_set = new Set(); // 检测有多少个源
let vue_ready = false;
// Background Script

// // host: 发送这个文件的url
// function wasmFound(data, host) {
//     // 用Web Crypto API 生成哈希作为filename
//     crypto.subtle.digest('SHA-256', new TextEncoder().encode(data)).then(async (hashBuffer) => {
//         const hashArray = Array.from(new Uint8Array(hashBuffer));
//         const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

//         if (!wasm_map.has(host)) {
//             wasm_map.set(host, []);
//         }

//         if (!host_set.has(host)) {
//             host_set.add(host)
//         }
//         let arr = wasm_map.get(host);

//         let wasm_obj = {
//             filename: hashHex,
//             content: data
//         }

//         arr.push(wasm_obj);
//     });

// }

function wasmFound(data, host, originalFilename) {
    crypto.subtle.digest('SHA-256', new TextEncoder().encode(data)).then(async (hashBuffer) => {
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

        if (!wasm_map.has(host)) {
            wasm_map.set(host, []);
        }

        if (!host_set.has(host)) {
            host_set.add(host)
        }
        let arr = wasm_map.get(host);

        let wasm_obj = {
            filename: hashHex,
            originalFilename: originalFilename, // 保存原始文件名
            content: data
        }

        arr.push(wasm_obj);
    });
}

function detect_wasm(host) {
    let data_send = {
        files: wasm_map.get(host),
        host: host
    }
    console.log(data_send)
    fetch(detect_server, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
            // 如果有需要的话，可以添加其他的请求头，比如认证令牌
            // "Authorization": "Bearer YOUR_TOKEN_HERE"
        },
        body: JSON.stringify(data_send) // 将JavaScript对象转换为JSON字符串
    })
        .then(response => {
            if (!response.ok) {
                // 网络响应状态码不为2xx的情况
                throw new Error(response);
            }
            return response.json(); // 解析JSON格式的响应数据
        })
        .then(data => {
            console.log("数据成功发送并接收到响应:", data);
            let temp = [];

            data.predictions.forEach((val) => {
                const key = Object.keys(val)[0];
                let obj = {
                    filename: key,
                    res: val[key],
                    note: val['note'],
                    host: host
                }
                temp.push(obj);
            });
            result_map.set(host, temp.slice());
            console.log(result_map)
            //send_detect_res(host)
            host_set.size === result_map.size && send_detect_res()
            // 在这里处理服务器返回的数据
        })
        .catch(error => {
            console.error("发送数据过程中出现问题:", error.message);
        })
        .finally(() => {
            wasm_map.set(host, []);
        })
    ;
}

function send_detect_res() {
    console.log("向vue页面发送数据")
    let data_arr = [];
    result_map.forEach((val, key) => {
        data_arr.push(...val);
    })
    const data_send = {
        data: data_arr,
        type: "data_to_show"
    }
    //chrome.runtime.sendMessage(data_send);
    chrome.tabs.query({active: true, currentWindow: true}, function (tabs) {
        var activeTab = tabs[0];
        chrome.tabs.sendMessage(activeTab.id, data_send);
    });
}

function download_wasm(base64Data, filename) {
    // 创建本地url
    const dataUrl = 'data:application/wasm;base64,' + base64Data;

    chrome.downloads.download({
        url: dataUrl,
        filename: filename,
        saveAs: true
    }, function (downloadId) {
        if (chrome.runtime.lastError) {
            console.error('下载失败:', chrome.runtime.lastError);
        } else {
            console.log('文件开始下载, 下载ID:', downloadId);
        }
    });
}

// 注意：这里假设数据已经是base64编码的字符串

// 监听注入的消息
chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
    if (changeInfo.status === 'complete' && tab.target) {
        // 适用于您想要的特定URLs
        console.log("注入成功")
    }
});

chrome.runtime.onMessage.addListener(async function (message, sender, sendResponse) {
    console.log("Message received:", message);

    // 通信测试
    if (message.greeting === "hello") {
        // 发送响应到content script
        sendResponse({farewell: "goodbye"});
    }
    if (message.type === "FROM_INJECTED_SCRIPT") {
        console.log("Message from injected script:", message.text);
        // 根据需要处理消息
    }

    // if (message.type === "WASM_FOUND") {
    //     const data = message.data; // Base64编码的WASM数据
    //     const host = message.host;
    //     // 在这里实现wasmFound的逻辑
    //     try {
    //         await wasmFound(data, host)
    //         // 这里应该发生个信息回去
    //         console.log('存储成功！')
    //     } catch {
    //         console.log('失败')
    //     }

    // }

    if (message.type === "WASM_FOUND") {
        const data = message.data; // Base64编码的WASM数据
        const host = message.host;
        const originalFilename = message.originalFilename; // 假设消息中包含原始文件名
        try {
            await wasmFound(data, host, originalFilename);
            console.log('存储成功！')
        } catch {
            console.log('失败')
        }
    }

    if (message.type === "WASM_COMPLETE") {
        // 所有wasm发送完毕
        console.log(`${message.host}总共有${message.count}个wasm`)
        console.log(wasm_map)
        detect_wasm(message.host)
        //console.log(result_map)

    }

    if (message.type === "vue_page_mounted") {
        console.log("vue页面已挂载")
        //wasm_map.size && send_detect_res(wasm_map)
        vue_ready = true;
    }

    if (message.action === 'closeTab') {
        console.log("快关闭网页！")
        chrome.tabs.query({active: true, currentWindow: true}, function (tabs) {
            chrome.tabs.remove(tabs[0].id);
        });
    }
    // 如果你需要异步响应，返回true
    // 这会保持消息通道开放，直到sendResponse被调用
    return true;
});
