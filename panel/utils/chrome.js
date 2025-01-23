const chrome = window.chrome;

export const sendMessage = (params) => {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tab) {
    chrome.tabs.sendMessage(tab[0].id, params, function (response) {
      alert("收到回复：" + response.state);
    });
  });
};

export const bookmarksListener = (cb) => {
  chrome.bookmarks.onTreeChanged.addListener(function (a) {
    cb(a);
  });
};

export const getBookmarks = () => {
  return new Promise((resolve) => {
    chrome.bookmarks.getTree(function (bookmarkTreeNodes) {
      resolve(bookmarkTreeNodes);
    });
  });
};

export function importBookmarks(bookmarkTreeNodes) {
  // 遍历书签树
  function traverseBookmarks(bookmarkNodes, parentId) {
    for (const node of bookmarkNodes) {
      // 如果节点是文件夹
      if (node.children) {
        // 创建一个新的文件夹节点
        chrome.bookmarks.create(
          {
            parentId: parentId,
            title: node.title,
          },
          function (newFolderNode) {
            // 递归遍历子节点
            traverseBookmarks(node.children, newFolderNode.id);
          }
        );
      }
      // 如果节点是书签
      else {
        // 创建一个新的书签节点
        chrome.bookmarks.create({
          parentId: parentId,
          title: node.title,
          url: node.url,
        });
      }
    }
  }

  // 从根节点开始遍历书签树
  traverseBookmarks(bookmarkTreeNodes[0].children, "1");
}
export function removeBookmarks(bookmarkTreeNodes) {
  // 遍历书签树，删除所有的书签
  function traverseBookmarks(bookmarkNodes) {
    for (const node of bookmarkNodes) {
      if (node.children) {
        traverseBookmarks(node.children);
      }
      // 删除书签节点
      chrome.bookmarks.removeTree(node.id);
    }
  }
  traverseBookmarks(bookmarkTreeNodes);
}

export function mergeBookmarks(bookmarks1, bookmarks2) {
  const root = {
    id: "0",
    title: "",
    children: [],
  };

  function findNode(parentNode, node) {
    return parentNode.children.find(
      (child) => child.title === node.title && child.url === node.url
    );
  }

  function mergeNodes(parentNode, nodes) {
    nodes.forEach((node) => {
      const existingNode = findNode(parentNode, node);
      if (existingNode) {
        if (node.children) {
          mergeNodes(existingNode, node.children);
        }
        // 合并节点属性
        Object.keys(node).forEach((key) => {
          if (key !== "children" && key !== "id") {
            existingNode[key] = node[key];
          }
        });
      } else {
        parentNode.children.push({ ...node });
      }
    });
  }

  [bookmarks1[0].children, bookmarks2[0].children].forEach((children) => {
    mergeNodes(root, children);
  });
  function compareNodes(node1, node2) {
    if (node1.children && !node2.children) {
      return -1; // node1在前
    } else if (!node1.children && node2.children) {
      return 1; // node2在前
    } else {
      return 0; // 顺序不变
    }
  }
  function sortChildren(parentNode) {
    if (parentNode.children) {
      parentNode.children.sort(compareNodes);

      parentNode.children.forEach((child) => {
        sortChildren(child); // 递归调用
      });
    }
  }

  sortChildren(root);

  return [root];
}
