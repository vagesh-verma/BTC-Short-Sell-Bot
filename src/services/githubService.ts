import { logger } from './loggerService';

export interface GitHubConfig {
  owner: string;
  repo: string;
  path: string;
  token: string;
}

export async function uploadToGitHub(config: GitHubConfig, fileName: string, content: string, message: string) {
  const { owner, repo, path, token } = config;
  const url = `https://api.github.com/repos/${owner}/${repo}/contents/${path}/${fileName}`;

  try {
    // Check if file exists to get its SHA (required for updates)
    let sha: string | undefined;
    const getResponse = await fetch(url, {
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json'
      }
    });

    if (getResponse.ok) {
      const data = await getResponse.json();
      sha = data.sha;
    }

    const body = {
      message,
      content: btoa(unescape(encodeURIComponent(content))), // Robust Base64 encode for UTF-8
      sha
    };

    const putResponse = await fetch(url, {
      method: 'PUT',
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });

    if (!putResponse.ok) {
      const errorData = await putResponse.json();
      throw new Error(errorData.message || 'Failed to upload to GitHub');
    }

    logger.success(`Successfully uploaded ${fileName} to GitHub.`);
  } catch (err) {
    logger.error(`GitHub upload error: ${err}`);
    throw err;
  }
}

export async function deleteFromGitHub(config: GitHubConfig, fileName: string, message: string) {
  const { owner, repo, path, token } = config;
  const url = `https://api.github.com/repos/${owner}/${repo}/contents/${path}/${fileName}`;

  try {
    // Must get SHA first
    const getResponse = await fetch(url, {
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json'
      }
    });

    if (!getResponse.ok) {
      if (getResponse.status === 404) {
        logger.info(`File ${fileName} not found on GitHub, skipping deletion.`);
        return;
      }
      const errorData = await getResponse.json();
      throw new Error(errorData.message || 'Failed to find file on GitHub');
    }

    const data = await getResponse.json();
    const sha = data.sha;

    const deleteResponse = await fetch(url, {
      method: 'DELETE',
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        sha
      })
    });

    if (!deleteResponse.ok) {
      const errorData = await deleteResponse.json();
      throw new Error(errorData.message || 'Failed to delete from GitHub');
    }

    logger.success(`Successfully deleted ${fileName} from GitHub.`);
  } catch (err) {
    logger.error(`GitHub deletion error: ${err}`);
    throw err;
  }
}

export async function fetchFromGitHub(config: GitHubConfig, fileName: string): Promise<string> {
  const { owner, repo, path, token } = config;
  const url = `https://api.github.com/repos/${owner}/${repo}/contents/${path}/${fileName}`;

  try {
    const response = await fetch(url, {
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json'
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || `Failed to fetch ${fileName} from GitHub`);
    }

    const data = await response.json();
    // GitHub returns content as base64
    return decodeURIComponent(escape(atob(data.content)));
  } catch (err) {
    logger.error(`GitHub fetch error: ${err}`);
    throw err;
  }
}
