<template>
  <div id="popup">
    <img alt="Vue logo" src="./assets/logo.png" class="logo" />
    <div class="hello">
      <p>Welcome to <span class="important">WebChecker</span>! Press <span class="important">Alt+V</span> to toggle the
        visibility of the popup window.</p>
    </div>
  </div>
</template>

<script>
import HelloWorld from "./components/HelloWorld.vue";

export default {
  name: "App",
  components: {
    HelloWorld,
  },
  data() {
    return {
      tableData: [],
    }
  }
};
</script>

<style>
#popup {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 10px;
}

.logo {
  width: 60px;
}

.hello {
  display: flex;
  justify-content: center;
}

.hello p {
  width: 250px;
  font-weight: 500;
  font-size: 16px;
  text-align: center;
  margin: 0;
}

.important {
  font-weight: 700;
}
</style>
