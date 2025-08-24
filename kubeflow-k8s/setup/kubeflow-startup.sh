#!/bin/sh

echo "Kernel parameters configuration"
sysctl fs.inotify.max_user_instances=2280
sysctl fs.inotify.max_user_watches=1255360

# Cluster config, creating cluster
cat <<EOF | kind create cluster --name=kubeflow --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  image: kindest-nvidia:1.0
  extraMounts:
    - hostPath: /dev/null
      containerPath: /var/run/nvidia-container-devices/all

  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    apiServer:
      extraArgs:
        "service-account-issuer": "https://kubernetes.default.svc"
        "service-account-signing-key-file": "/etc/kubernetes/pki/sa.key"
EOF

# Export kubeconfig, context setting
echo "Saving kubeconfig for the cluster to /home/uros/Master/SIR-Kubeflow/kubeflow-config"
kind get kubeconfig --name kubeflow > /home/uros/Master/SIR-Kubeflow/kubeflow-config.yaml
export KUBECONFIG=/home/uros/Master/SIR-Kubeflow/kubeflow-config.yaml
chmod a+w /home/uros/Master/SIR-Kubeflow/kubeflow-config.yaml
echo $KUBECONFIG

docker login
kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=$HOME/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson

# Create nvidia-device-plugin deamonset to use GPU
cd /home/uros/Master/SIR-Kubeflow
kubectl apply -f nvidia-device-plugin.yaml
sleep 60

# Kubeflow deploy
echo "Creating Kubeflow objects."
cd /home/uros/Master/SIR-Kubeflow/manifests-1.10.1
while ! kubectl kustomize example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 60; done
