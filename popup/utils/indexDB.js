export class IndexedDB {
  constructor(databaseName, storeName) {
    this.databaseName = databaseName;
    this.storeName = storeName;
    this.db = null;
  }

  open() {
    return new Promise((resolve, reject) => {
      const request = window.indexedDB.open(this.databaseName);

      request.onerror = () => {
        reject(new Error("Failed to open database"));
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        this.db = event.target.result;
        if (!this.db.objectStoreNames.contains(this.storeName)) {
          this.db.createObjectStore(this.storeName, {
            keyPath: "id",
            autoIncrement: true,
          });
        }
      };
    });
  }

  createDatabase() {
    return new Promise((resolve, reject) => {
      const request = window.indexedDB.open(this.databaseName);

      request.onerror = () => {
        reject(new Error("Failed to create database"));
      };

      request.onsuccess = () => {
        this.db = request.result;
        this.db.close();
        resolve();
      };

      request.onupgradeneeded = (event) => {
        this.db = event.target.result;
        if (!this.db.objectStoreNames.contains(this.storeName)) {
          this.db.createObjectStore(this.storeName, {
            keyPath: "id",
            autoIncrement: true,
          });
        }
        this.db.close();
        resolve();
      };
    });
  }

  close() {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }

  add(data) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(this.storeName, "readwrite");
      const objectStore = transaction.objectStore(this.storeName);
      const request = objectStore.add(data);

      request.onsuccess = () => {
        resolve(request.result);
      };

      request.onerror = () => {
        reject(new Error("Failed to add data"));
      };
    });
  }

  getAll() {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(this.storeName, "readonly");
      const objectStore = transaction.objectStore(this.storeName);
      const request = objectStore.getAll();

      request.onsuccess = () => {
        resolve(request.result);
      };

      request.onerror = () => {
        reject(new Error("Failed to get data"));
      };
    });
  }

  getById(id) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(this.storeName, "readonly");
      const objectStore = transaction.objectStore(this.storeName);
      const request = objectStore.get(id);

      request.onsuccess = () => {
        resolve(request.result);
      };

      request.onerror = () => {
        reject(new Error("Failed to get data"));
      };
    });
  }

  delete(id) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(this.storeName, "readwrite");
      const objectStore = transaction.objectStore(this.storeName);
      const request = objectStore.delete(id);

      request.onsuccess = () => {
        resolve();
      };

      request.onerror = () => {
        reject(new Error("Failed to delete data"));
      };
    });
  }

  update(id, newData) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(this.storeName, "readwrite");
      const objectStore = transaction.objectStore(this.storeName);
      const getRequest = objectStore.get(id);

      getRequest.onsuccess = () => {
        const oldData = getRequest.result;

        if (!oldData) {
          const addRequest = objectStore.add({ ...newData, id });

          addRequest.onsuccess = () => {
            resolve({ ...newData, id });
          };

          addRequest.onerror = () => {
            reject(new Error("Failed to add data"));
          };
        } else {
          const mergedData = { ...oldData, ...newData };
          const putRequest = objectStore.put(mergedData);

          putRequest.onsuccess = () => {
            resolve(mergedData);
          };

          putRequest.onerror = () => {
            reject(new Error("Failed to update data"));
          };
        }
      };

      getRequest.onerror = () => {
        reject(new Error("Failed to get data"));
      };
    });
  }
}
