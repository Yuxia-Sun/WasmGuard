<template>
  <div class="hello">
    <h1 id="message">你好{{ inputVal }}</h1>
    <input id="input1" v-model="inputVal" type="text" />
   
  </div>
</template>

<script>
import { sendMessage } from "../../utils/chrome.js";
export default {
  name: "HelloWorld",
  props: {
    msg: String,
  },
  data() {
    return {
      inputVal: "",
    };
  },
  methods: {
    sendData() {
      sendMessage({ action: "sendData", data: this.inputVal });
    },
  },
};
</script>

<style scoped></style>
