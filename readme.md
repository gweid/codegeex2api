## codegeex2api

codegeex to api 项目



### 部署


下载项目

```shell
https://github.com/gweid/codegeex2api.git
```

进入项目

```shell
cd codegeex2api
```



创建 codegeex.txt，里面填入 code-token（登录: https://codegeex.cn，抓包请求，在请求头中获取）

```shell
vim codegeex.txt
```



创建 client_api_keys.json，里面是你的秘钥，用于 cherry studio 等连接使用

```shell
vim client_api_keys.json
```



启动

```shell
docker-compose up -d
```



### 使用

- 获取模型：http://127.0.0.1:3005/v1/models

- 发送消息：http://127.0.0.1:3005/v1/chat/message
