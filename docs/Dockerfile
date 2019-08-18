FROM node:10
WORKDIR /usr/src/app
COPY package*.json ./
COPY . /usr/src/app
RUN npm install
RUN yarn global add vuepress@next axios vue-intercom vue-analytics
RUN vuepress build
