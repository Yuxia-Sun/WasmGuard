(function () {

    // 发布订阅板块
    const host = document.location.hostname; // 待修改，传送时可能需要附带
    let total = 0;
    let activeWasmOperations = 0;
    const wasmOperationsCompletedCallbacks = new Set(); // callback queue
    let checkCompletionTimeout;

    // 调用到wasmFound的时候顺便调用
    function onWasmOperationStarted() {
        activeWasmOperations++;
        total++;
    }

    function sendComplete() {
        window.postMessage({ type: "WASM_COMPLETE", count: total, host: host }, "*");
    }

    function onWasmOperationCompleted() {
        activeWasmOperations--;
        clearTimeout(checkCompletionTimeout);
        if (activeWasmOperations === 0) {
            // 防抖
            checkCompletionTimeout = setTimeout(() => {
                // 这里执行所有操作真正完成后的逻辑
                wasmOperationsCompletedCallbacks.forEach(callback => callback());
                wasmOperationsCompletedCallbacks.clear(); // 清空回调列表
            }, 1000); // 延迟时间可以根据实际情况调整
        }
    }

    function onAllWasmOperationsCompleted(callback) {
        if (activeWasmOperations === 0) {
            callback();
        } else {
            wasmOperationsCompletedCallbacks.add(callback);
        }
    }

    function wasmFound(data) {
        onWasmOperationStarted(); // 异步操作开始

        // 使用Promise处理异步逻辑
        new Promise((resolve, reject) => {
            // 执行异步操作，比如通过postMessage发送数据
            window.postMessage({ type: "WASM_FOUND", data: data, host: host }, "*");

            // 假设我们知道何时postMessage成功完成
            // 这里直接调用resolve()，在实际应用中可能需要基于事件或回调来调用
            resolve();
        })
            .catch((error) => {
                // 处理异步操作中的错误
                console.error("Error in wasmFound:", error);
            })
            .finally(() => {
                // await 录入完毕的信息
                // setTimeout(() => {
                //     onWasmOperationCompleted();
                // },1000)
                // 无论成功还是失败，标记异步操作完成
            });
    }


    function bufferToBase64(buffer) {
        let binary = "";
        let bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    (function () {
        let old = {};
        function wrap(name) {
            old[name] = WebAssembly[name];
            WebAssembly[name] = function (bufferSource) {
                wasmFound(bufferToBase64(bufferSource));
                onAllWasmOperationsCompleted(sendComplete)
                let result = old[name].call(WebAssembly, ...arguments);
                setTimeout(() => {
                    onWasmOperationCompleted();
                }, 1000)
                return result;
            };
        }
        wrap("instantiate");
        wrap("compile");
    })();

    WebAssembly.instantiateStreaming = async function (source, importObject) {
        let response = await source;
        let body = await response.arrayBuffer();
        return WebAssembly.instantiate(body, importObject);
    };

    WebAssembly.compileStreaming = async function (source) {
        let response = await source;
        let body = await response.arrayBuffer();
        return WebAssembly.compile(body);
    };

    const handler = {
        construct(target, args) {
            wasmFound(bufferToBase64(args[0]));
            onAllWasmOperationsCompleted(sendComplete)

            let result = new target(...args);
            setTimeout(() => {
                onWasmOperationCompleted();
            }, 1000)
            return result;
        }
    };
    WebAssembly.Module = new Proxy(WebAssembly.Module, handler);
    // 发送消息到网页，以便content script能够监听到
    window.postMessage({ type: "FROM_INJECTED_SCRIPT", text: "Hello from injected script!" }, "*");

})();
