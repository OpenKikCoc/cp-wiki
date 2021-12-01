FROM node:slim

RUN npm i docsify-cli -g

COPY . /docs

CMD docsify serve /docs
