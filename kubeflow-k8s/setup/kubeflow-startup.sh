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
  image: kindest-nvidia:1.1
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

BASE_DIR=/home/uros/Master/SIR-Kubeflow/kubeflow-k8s/setup
echo "Saving kubeconfig for the cluster to ${BASE_DIR}/kubeflow-config"

kind get kubeconfig --name kubeflow > "$BASE_DIR"/kubeflow-config.yaml
export KUBECONFIG=/home/uros/Master/SIR-Kubeflow/kubeflow-k8s/setup/kubeflow-config.yaml
chmod a+w "$BASE_DIR"/kubeflow-config.yaml

docker login
kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=$HOME/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson

# Create nvidia-device-plugin deamonset to use GPU
kubectl apply -f "$BASE_DIR"/device-plugins/nvidia-device-plugin.yaml
sleep 60

# Kubeflow deploy
echo "Creating Kubeflow objects."
cd "$BASE_DIR"/manifests-1.11.0
while ! kubectl kustomize example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 60; done

# Training manager, training runtimes
sleep 60
kubectl apply --server-side -f "$BASE_DIR"/cluster-training-runtimes/

# RBAC, configure permissions for default-editor service account
kubectl apply  -f "$BASE_DIR"/rbac/

# Creating private ssh secret and pod defaults - in order for KFP and Git to be available inside notebook pods
kubectl create secret generic git-ssh-key --from-file=id_ed25519=/home/uros/.ssh/id_ed25519 --from-file=known_hosts=/home/uros/.ssh/known_hosts -n kubeflow-user-example-com
kubectl apply -f "$BASE_DIR"/poddefault-pipelines-token.yaml
kubectl apply -f "$BASE_DIR"/poddefault-git-ssh-key.yaml

kubectl apply -f "$BASE_DIR"/services/

# This fixes a login bug that started to occur "Jwt verification fails"
#kubectl patch sidecar default -n istio-system --type='json' -p='[{"op": "add", "path": "/spec/egress/0/hosts/-", "value": "auth/*"}]'
kubectl patch requestauthentication dex-jwt -n istio-system --type='json' -p='[{"op": "add", "path": "/spec/jwtRules/0/jwksUri", "value": "http://dex.auth.svc.cluster.local:5556/dex/keys"}]'
kubectl set env deployment/istiod -n istio-system PILOT_JWT_ENABLE_REMOTE_JWKS=envoy
sleep 5
kubectl rollout restart deployment/istiod -n istio-system
kubectl rollout restart deployment/istio-ingressgateway -n istio-system

sleep 20
echo "Cluster created successfully. Kubeflow is up and running..."
