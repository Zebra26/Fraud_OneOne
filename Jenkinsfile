pipeline {
    agent any

    environment {
        PYTHON_VERSION = "3.11"
        COMPOSE_FILE = "docker-compose.dev.yml"
    }

    options {
        timestamps()
        ansiColor('xterm')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python') {
            steps {
                withEnv(["VIRTUAL_ENV=${env.WORKSPACE}/.venv", "PATH=${env.WORKSPACE}/.venv/Scripts:${env.WORKSPACE}/.venv/bin:${env.PATH}"]) {
                    sh '''
                        python -m venv .venv
                        . .venv/bin/activate
                        pip install --upgrade pip
                        pip install -r backend/requirements.txt
                        pip install -r services/ml-inference/requirements.txt
                    '''
                }
            }
        }

        stage('Static Checks') {
            steps {
                withEnv(["VIRTUAL_ENV=${env.WORKSPACE}/.venv", "PATH=${env.WORKSPACE}/.venv/Scripts:${env.WORKSPACE}/.venv/bin:${env.PATH}"]) {
                    sh '''
                        . .venv/bin/activate
                        python -m compileall backend services/ml-inference services/trainer
                    '''
                }
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    docker compose -f ${COMPOSE_FILE} build backend
                    docker compose -f ${COMPOSE_FILE} run --rm backend pytest
                '''
            }
            post {
                always {
                    sh 'docker compose -f ${COMPOSE_FILE} down --remove-orphans || true'
                }
            }
        }

        stage('Train Models') {
            steps {
                sh '''
                    docker compose -f ${COMPOSE_FILE} build trainer
                    docker compose -f ${COMPOSE_FILE} run --rm trainer python train_pipeline.py
                '''
            }
            post {
                always {
                    sh 'docker compose -f ${COMPOSE_FILE} down --remove-orphans || true'
                }
            }
        }

        stage('Package Images') {
            steps {
                sh '''
                    docker compose -f ${COMPOSE_FILE} build backend ml-inference
                '''
            }
        }
    }

    post {
        success {
            archiveArtifacts artifacts: 'models/**', onlyIfSuccessful: true
        }
        cleanup {
            sh 'docker system prune -f || true'
        }
    }
}
