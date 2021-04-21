FROM mo5mami/torchserve_env
MAINTAINER mami.mokhtar123@gmail.com
RUN apt update && apt install --no-install-recommends -y openjdk-11-jre-headless
COPY ./ /serve
WORKDIR /serve
EXPOSE 80
#ENTRYPOINT ["/bin/sh" , "-c" , "./run_prod"]
CMD ["bash"]